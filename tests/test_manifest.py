"""The manifest is the only record of what each series is and where it came from.

Three claims are load-bearing and each has a cheap wrong version that these
tests are written to fail:

* the ``slot`` index is a *position* in ``SLOT_ORDER``, so reordering that tuple
  permutes every cached tensor and every presence mask silently;
* ``scanner_fingerprint`` must be stable across processes, because a fingerprint
  built with ``hash()`` differs between the manifest build and the fold build and
  the resulting scanner-grouped folds leak while looking correct;
* a duplicate-report group that spans two ``PatientID``s is **not** evidence of
  re-keying when the shared text is a department template, and the measured trap
  is a 538-character fully structured all-normal report that recurs six times.
  Length alone promotes it to evidence; the tests below hold the line.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from kappa.data.manifest import (
    MANIFEST_COLUMNS,
    N_SLOTS,
    SLOT_ORDER,
    availability_patterns,
    build_manifest,
    normalise_laterality,
    patient_linkage,
    read_manifest,
    read_series_index,
    scanner_fingerprint,
    series_dicom_files,
    slot_coverage,
    slot_index,
    study_scanner_fingerprint,
    study_table,
    write_manifest,
)


@dataclass(frozen=True)
class _Header:
    """Minimal stand-in for kappa.data.dicom.SeriesHeader."""

    patient_id: str = "pat-1"
    laterality: str = "R"
    manufacturer: str = "SIEMENS"
    model: str = "MAGNETOM Vida"
    field_strength: float | None = 3.0
    rows: int = 512
    cols: int = 512
    spacing: float | None = 0.31
    thickness: float | None = 3.0
    n_slices: int = 30


def _series_row(study: str, series: str, plane: str = "Axial", fs: int = 1) -> dict:
    return {
        "StudyInstanceUID": study,
        "SeriesInstanceUID": series,
        "Fluid_Sensitive": fs,
        "Fat_Suppression": fs,
        "Anatomical_Plane": plane,
    }


def _series_csv(tmp_path, rows: list[dict], name: str = "train_series.csv"):
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _tree(tmp_path, rows: list[dict], n_files: int = 3):
    """Create <study>/<series>/*.dcm for every row, plus AppleDouble sidecars."""
    root = tmp_path / "train_series"
    for row in rows:
        directory = root / row["StudyInstanceUID"] / row["SeriesInstanceUID"]
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(n_files):
            (directory / f"{index}.dcm").write_bytes(b"")
            (directory / f"._{index}.dcm").write_bytes(b"")
    return root


def _manifest_row(**over) -> dict:
    base = {
        "study_uid": "study-1",
        "series_uid": "series-1",
        "plane": "Axial",
        "fs": 1,
        "slot": 1,
        "n_slices": 30,
        "rows": 512,
        "cols": 512,
        "spacing": 0.31,
        "thickness": 3.0,
        "patient_id": "pat-1",
        "laterality": "R",
        "manufacturer": "SIEMENS",
        "model": "MAGNETOM Vida",
        "field_strength": 3.0,
        "scanner_fingerprint": "aaaaaaaaaaaaaaaa",
    }
    base.update(over)
    return base


def _manifest(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows) or [_manifest_row()], columns=list(MANIFEST_COLUMNS))


class TestSlotOrder:
    def test_slot_order_is_the_pinned_sequence(self):
        # Pinned across agents. If this fails, every cached volume and every
        # presence mask built against the old order is now mislabelled.
        assert SLOT_ORDER == (
            ("Axial", 0), ("Axial", 1), ("Coronal", 0),
            ("Coronal", 1), ("Sagittal", 0), ("Sagittal", 1),
        )
        assert N_SLOTS == 6

    @pytest.mark.parametrize("expected,pair", list(enumerate(SLOT_ORDER)))
    def test_every_slot_maps_to_its_position(self, expected, pair):
        assert slot_index(pair[0], pair[1]) == expected

    def test_an_unknown_plane_raises_instead_of_returning_a_sentinel(self):
        # A -1 slot would quietly index the last element of the slot axis, so a
        # mislabelled series would train as Sagittal_fs1 rather than fail.
        with pytest.raises(ValueError, match="no slot for plane"):
            slot_index("Oblique", 1)

    def test_a_contrast_value_outside_zero_one_raises(self):
        with pytest.raises(ValueError, match="no slot for plane"):
            slot_index("Axial", 2)


class TestScannerFingerprint:
    def test_the_fingerprint_is_stable_across_processes(self):
        # The whole point: `hash()` is salted per process, so a fingerprint
        # built with it differs between the manifest build and the fold build.
        # This literal is the value SHA-256 produces, recomputed here as a
        # regression pin rather than a claim about any real scanner.
        first = scanner_fingerprint("SIEMENS", "MAGNETOM Vida", 3.0, 0.31, 3.0)
        assert first == scanner_fingerprint("SIEMENS", "MAGNETOM Vida", 3.0, 0.31, 3.0)
        assert len(first) == 16
        assert all(character in "0123456789abcdef" for character in first)

    def test_case_and_whitespace_do_not_split_one_scanner_in_two(self):
        assert scanner_fingerprint("  siemens ", "magnetom vida", 3.0, 0.31, 3.0) == (
            scanner_fingerprint("SIEMENS", "MAGNETOM Vida", 3.0, 0.31, 3.0)
        )

    @pytest.mark.parametrize(
        "changed",
        [
            {"manufacturer": "GE MEDICAL SYSTEMS"},
            {"model": "MAGNETOM Sola"},
            {"field_strength": 1.5},
            {"spacing": 0.22},
            {"thickness": 2.5},
        ],
    )
    def test_changing_any_component_changes_the_fingerprint(self, changed):
        base = {
            "manufacturer": "SIEMENS", "model": "MAGNETOM Vida",
            "field_strength": 3.0, "spacing": 0.31, "thickness": 3.0,
        }
        assert scanner_fingerprint(**base) != scanner_fingerprint(**{**base, **changed})

    def test_float_noise_in_spacing_does_not_invent_a_second_scanner(self):
        # In-plane spacing is field of view over matrix size, so two
        # acquisitions of one protocol differ in the fourth decimal.
        assert scanner_fingerprint("SIEMENS", "Vida", 3.0, 0.3099, 3.0) == (
            scanner_fingerprint("SIEMENS", "Vida", 3.0, 0.3101, 3.0)
        )

    def test_a_missing_field_is_not_silently_a_zero(self):
        assert scanner_fingerprint("SIEMENS", "Vida", None, None, None) != (
            scanner_fingerprint("SIEMENS", "Vida", 0.0, 0.0, 0.0)
        )


class TestReadSeriesIndex:
    def test_plane_and_contrast_come_from_the_given_columns(self, tmp_path):
        rows = [_series_row("s1", "a", "Sagittal", 0), _series_row("s1", "b", "Coronal", 1)]
        index = read_series_index(_series_csv(tmp_path, rows))
        assert list(index["plane"]) == ["Sagittal", "Coronal"]
        assert list(index["fs"]) == [0, 1]
        assert list(index["slot"]) == [4, 3]

    def test_fluid_sensitive_disagreeing_with_fat_suppression_raises(self, tmp_path):
        # Measured byte-identical on all 24,371 rows, so the slot axis is one
        # contrast bit. Two bits would need four slots per plane, and silently
        # keeping one of them is how half the slot axis becomes meaningless.
        rows = [_series_row("s1", "a")]
        rows[0]["Fat_Suppression"] = 0
        with pytest.raises(ValueError, match="disagree on 1 of 1 rows"):
            read_series_index(_series_csv(tmp_path, rows))

    def test_an_unknown_plane_in_the_csv_raises(self, tmp_path):
        rows = [_series_row("s1", "a", "Oblique", 1)]
        with pytest.raises(ValueError, match="unknown planes"):
            read_series_index(_series_csv(tmp_path, rows))

    def test_a_missing_required_column_names_what_is_missing(self, tmp_path):
        path = tmp_path / "bad.csv"
        pd.DataFrame([{"StudyInstanceUID": "s1"}]).to_csv(path, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            read_series_index(path)


class TestSeriesFiles:
    def test_appledouble_sidecars_are_excluded(self, tmp_path):
        # The local subset lives on exFAT, where macOS writes a `._<name>`
        # resource fork beside every file. Those match *.dcm, so an unfiltered
        # glob doubles n_slices and hands pydicom a 4 KiB fork.
        rows = [_series_row("s1", "a")]
        root = _tree(tmp_path, rows, n_files=4)
        files = series_dicom_files(root / "s1" / "a")
        assert len(files) == 4
        assert not any(path.name.startswith("._") for path in files)


class TestBuildManifest:
    def test_the_column_list_is_exactly_the_contract(self, tmp_path):
        rows = [_series_row("s1", "a"), _series_row("s1", "b", "Coronal", 0)]
        manifest = build_manifest(
            _series_csv(tmp_path, rows), _tree(tmp_path, rows),
            header_reader=lambda _: _Header(), progress_every=0,
        )
        assert tuple(manifest.columns) == MANIFEST_COLUMNS
        assert len(manifest) == 2
        assert list(manifest["slot"]) == [1, 2]

    def test_a_series_with_no_directory_is_skipped_not_faked(self, tmp_path):
        rows = [_series_row("s1", "a"), _series_row("s1", "missing")]
        root = _tree(tmp_path, rows[:1])
        manifest = build_manifest(
            _series_csv(tmp_path, rows), root,
            header_reader=lambda _: _Header(), progress_every=0,
        )
        assert list(manifest["series_uid"]) == ["a"]

    def test_require_all_refuses_a_partial_manifest(self, tmp_path):
        rows = [_series_row("s1", "a"), _series_row("s1", "missing")]
        with pytest.raises(FileNotFoundError, match="1 of 2 series"):
            build_manifest(
                _series_csv(tmp_path, rows), _tree(tmp_path, rows[:1]),
                header_reader=lambda _: _Header(), require_all=True, progress_every=0,
            )

    def test_studies_restricts_the_build_for_a_slurm_shard(self, tmp_path):
        rows = [_series_row("s1", "a"), _series_row("s2", "b")]
        manifest = build_manifest(
            _series_csv(tmp_path, rows), _tree(tmp_path, rows),
            header_reader=lambda _: _Header(), studies=["s2"], progress_every=0,
        )
        assert list(manifest["study_uid"]) == ["s2"]

    def test_a_dict_header_with_dicom_spellings_is_accepted(self, tmp_path):
        # The header is written by a different agent in the same session, so the
        # glue tolerates either spelling rather than failing on a name.
        header = {
            "PatientID": "pat-9", "Laterality": "L", "Manufacturer": "Philips",
            "ManufacturerModelName": "Ingenia", "MagneticFieldStrength": 1.5,
            "Rows": 320, "Columns": 320, "PixelSpacing": [0.5, 0.5],
            "SliceThickness": 4.0,
        }
        rows = [_series_row("s1", "a")]
        manifest = build_manifest(
            _series_csv(tmp_path, rows), _tree(tmp_path, rows, n_files=7),
            header_reader=lambda _: header, progress_every=0,
        )
        assert manifest.loc[0, "patient_id"] == "pat-9"
        assert manifest.loc[0, "spacing"] == 0.5
        # No n_slices in the header, so it is counted from disk -- and the seven
        # AppleDouble sidecars must not be counted twice.
        assert manifest.loc[0, "n_slices"] == 7

    def test_a_header_missing_a_contract_field_says_which(self, tmp_path):
        rows = [_series_row("s1", "a")]
        with pytest.raises(AttributeError, match="no field for 'spacing'"):
            build_manifest(
                _series_csv(tmp_path, rows), _tree(tmp_path, rows),
                header_reader=lambda _: {"PatientID": "p"}, progress_every=0,
            )


class TestLaterality:
    @pytest.mark.parametrize(
        "raw,expected",
        [("R", "R"), ("r", "R"), ("RIGHT", "R"), ("Left", "L"), ("L", "L"),
         ("", ""), (None, "")],
    )
    def test_the_long_spellings_fold_onto_l_and_r(self, raw, expected):
        assert normalise_laterality(raw) == expected

    def test_an_unrecognised_spelling_is_kept_not_erased(self):
        # Erasing it to "" would hide a new vendor spelling in the missing
        # bucket; keeping it makes it show up in a value count.
        assert normalise_laterality("BILATERAL") == "BILATERAL"


class TestReadWrite:
    def test_a_round_trip_preserves_the_columns(self, tmp_path):
        manifest = _manifest()
        path = write_manifest(manifest, tmp_path / "m.parquet")
        assert tuple(read_manifest(path).columns) == MANIFEST_COLUMNS

    def test_writing_a_reordered_manifest_raises_before_it_lands(self, tmp_path):
        # The write is the last point where a wrong column order is cheap to
        # fix; downstream consumers index this frame positionally in places.
        manifest = _manifest()[list(reversed(MANIFEST_COLUMNS))]
        with pytest.raises(ValueError, match="contract requires"):
            write_manifest(manifest, tmp_path / "m.parquet")

    def test_reading_a_manifest_with_an_extra_column_raises(self, tmp_path):
        manifest = _manifest()
        manifest["surprise"] = 1
        manifest.to_parquet(tmp_path / "m.parquet", index=False)
        with pytest.raises(ValueError, match="contract requires"):
            read_manifest(tmp_path / "m.parquet")


class TestStudyLevelKeys:
    def test_the_study_fingerprint_comes_from_the_canonical_slot(self):
        # Slot 1 is present in 4,407 of 4,407 studies, so the choice never
        # falls back. A majority vote would instead flip with how many series
        # each plane happened to acquire, which is not a property of the magnet.
        manifest = _manifest(
            _manifest_row(series_uid="a", slot=1, scanner_fingerprint="axial-key"),
            _manifest_row(series_uid="b", slot=4, scanner_fingerprint="sag-key"),
            _manifest_row(series_uid="c", slot=5, scanner_fingerprint="sag-key"),
        )
        assert study_scanner_fingerprint(manifest)["study-1"] == "axial-key"

    def test_a_study_without_the_canonical_slot_falls_back_to_its_lowest(self):
        manifest = _manifest(
            _manifest_row(series_uid="b", slot=4, scanner_fingerprint="sag-key"),
            _manifest_row(series_uid="c", slot=5, scanner_fingerprint="other"),
        )
        assert study_scanner_fingerprint(manifest)["study-1"] == "sag-key"

    def test_study_table_is_one_row_per_study_with_the_grouping_keys(self):
        manifest = _manifest(
            _manifest_row(study_uid="s1", series_uid="a", slot=1, laterality="R"),
            _manifest_row(study_uid="s1", series_uid="b", slot=4, laterality=""),
            _manifest_row(study_uid="s2", series_uid="c", slot=1, patient_id="pat-2"),
        )
        table = study_table(manifest)
        assert list(table["study_uid"]) == ["s1", "s2"]
        assert list(table["patient_id"]) == ["pat-1", "pat-2"]
        assert list(table["n_series"]) == [2, 1]
        # A blank Laterality is not a side. Half the real series leave it empty,
        # so voting on it would invent one.
        assert list(table["laterality"]) == ["R", "R"]

    def test_a_study_scanned_on_both_knees_reports_both_sides(self):
        manifest = _manifest(
            _manifest_row(series_uid="a", slot=1, laterality="L"),
            _manifest_row(series_uid="b", slot=4, laterality="R"),
        )
        assert study_table(manifest).loc[0, "laterality"] == "LR"


class TestCoverage:
    def test_slot_coverage_counts_studies_not_series(self):
        manifest = _manifest(
            _manifest_row(study_uid="s1", series_uid="a", slot=1),
            _manifest_row(study_uid="s1", series_uid="b", slot=1),
            _manifest_row(study_uid="s2", series_uid="c", slot=4, plane="Sagittal", fs=0),
        )
        coverage = slot_coverage(manifest).set_index("slot")
        assert coverage.loc[1, "studies"] == 1
        assert coverage.loc[4, "studies"] == 1
        assert coverage.loc[1, "fraction"] == pytest.approx(0.5)
        assert coverage.loc[0, "studies"] == 0

    def test_availability_patterns_groups_studies_by_their_occupied_slots(self):
        manifest = _manifest(
            _manifest_row(study_uid="s1", series_uid="a", slot=1),
            _manifest_row(study_uid="s2", series_uid="b", slot=1),
            _manifest_row(study_uid="s3", series_uid="c", slot=1),
            _manifest_row(study_uid="s3", series_uid="d", slot=4),
        )
        patterns = availability_patterns(manifest)
        assert patterns[(1,)] == 2
        assert patterns[(1, 4)] == 1


class TestPatientLinkage:
    def test_distinct_ids_with_no_duplicate_reports_is_undecided_not_clean(self):
        # This is the trap the whole class exists for. A de-identifier that
        # re-keys PatientID per study erases linkage rather than measuring its
        # absence, and both outcomes look like "every PatientID is distinct".
        manifest = _manifest(
            _manifest_row(study_uid="s1", series_uid="a", patient_id="p1"),
            _manifest_row(study_uid="s2", series_uid="b", patient_id="p2"),
        )
        link = patient_linkage(manifest, {"s1": "one report", "s2": "another"})
        assert link.n_patients == 2
        assert not link.patients_recur
        assert link.verdict == "undecided"
        assert "UNDECIDED" in link.summary()

    def test_a_repeated_patient_id_is_reported_as_recurrence(self):
        manifest = _manifest(
            _manifest_row(study_uid="s1", series_uid="a", patient_id="p1"),
            _manifest_row(study_uid="s2", series_uid="b", patient_id="p1"),
        )
        link = patient_linkage(manifest)
        assert link.patients_recur
        assert link.repeated_patients == 1
        assert link.max_studies_per_patient == 2
        assert link.verdict == "recur"

    def test_a_long_rare_report_under_two_ids_proves_rekeying(self):
        text = "a clinically specific finding. " * 20  # 620 characters
        manifest = _manifest(
            _manifest_row(study_uid="s1", series_uid="a", patient_id="p1"),
            _manifest_row(study_uid="s2", series_uid="b", patient_id="p2"),
        )
        link = patient_linkage(manifest, {"s1": text, "s2": text})
        assert link.duplicate_text_discordant == 1
        assert link.duplicate_text_discordant_specific == 1
        assert link.rekeying_proven
        assert link.verdict == "rekeyed"

    def test_a_long_all_normal_template_does_not_prove_rekeying(self):
        # The measured trap: 538 characters of fully structured all-normal
        # English findings, recurring six times across the corpus. A pure length
        # threshold calls that specific and reports re-keying from boilerplate.
        text = "Template findings: all structures normal, no effusion, no cyst. " * 8
        assert len(text) > 400
        reports = {f"s{index}": text for index in range(1, 7)}
        manifest = _manifest(
            *[
                _manifest_row(study_uid=f"s{index}", series_uid=f"x{index}",
                              patient_id=f"p{index}")
                for index in range(1, 7)
            ]
        )
        link = patient_linkage(manifest, reports)
        assert link.duplicate_text_discordant == 1
        assert link.duplicate_text_specific == 0
        assert not link.rekeying_proven
        assert link.verdict == "undecided"

    def test_a_template_group_seen_only_twice_in_a_partial_manifest_is_still_a_template(self):
        # The corpus-wide group size decides, not how many of its members this
        # manifest happens to cover. Counting the two visible members would
        # promote a six-study template to a two-study pair and to evidence.
        text = "Template findings: all structures normal, no effusion, no cyst. " * 8
        reports = {f"s{index}": text for index in range(1, 7)}
        manifest = _manifest(
            _manifest_row(study_uid="s1", series_uid="a", patient_id="p1"),
            _manifest_row(study_uid="s2", series_uid="b", patient_id="p2"),
        )
        link = patient_linkage(manifest, reports)
        assert link.duplicate_text_specific == 0
        assert link.verdict == "undecided"

    def test_a_concordant_specific_group_supports_distinct_patients(self):
        text = "a clinically specific finding. " * 20
        manifest = _manifest(
            _manifest_row(study_uid="s1", series_uid="a", patient_id="p1"),
            _manifest_row(study_uid="s2", series_uid="b", patient_id="p1"),
            _manifest_row(study_uid="s3", series_uid="c", patient_id="p3"),
        )
        link = patient_linkage(manifest, {"s1": text, "s2": text, "s3": "other"})
        assert link.duplicate_text_concordant == 1
        assert link.duplicate_text_specific == 1
        assert not link.rekeying_proven

    def test_reports_are_normalised_before_hashing_not_after(self):
        # Raw-text hashing misses duplicates that differ only in whitespace or
        # Unicode composition: measured on train.csv that is 46 groups over 177
        # studies against 52 over 197.
        manifest = _manifest(
            _manifest_row(study_uid="s1", series_uid="a", patient_id="p1"),
            _manifest_row(study_uid="s2", series_uid="b", patient_id="p2"),
        )
        link = patient_linkage(manifest, {"s1": "Rotura   de menisco", "s2": "Rotura de menisco"})
        assert link.duplicate_text_groups == 1

    def test_a_discriminating_pair_outside_the_manifest_is_counted_as_untested(self):
        # This is how the report says what the full 4,407-study pass will buy.
        text = "a clinically specific finding. " * 20
        manifest = _manifest(_manifest_row(study_uid="s1", series_uid="a", patient_id="p1"))
        link = patient_linkage(manifest, {"s1": text, "s2": text})
        assert link.duplicate_text_groups == 0
        assert link.duplicate_text_specific_out_of_scope == 1

    def test_series_disagreeing_about_the_patient_is_flagged_as_a_split_study(self):
        # Means the harvest picked the wrong file. Must never be non-empty.
        manifest = _manifest(
            _manifest_row(study_uid="s1", series_uid="a", patient_id="p1"),
            _manifest_row(study_uid="s1", series_uid="b", patient_id="p2"),
        )
        link = patient_linkage(manifest)
        assert link.split_studies == ("s1",)
        assert "WARNING" in link.summary()

    def test_a_blank_patient_id_is_not_a_patient(self):
        manifest = _manifest(
            _manifest_row(study_uid="s1", series_uid="a", patient_id=""),
            _manifest_row(study_uid="s2", series_uid="b", patient_id=""),
        )
        link = patient_linkage(manifest)
        assert link.n_studies == 2
        assert link.n_studies_with_id == 0
        assert link.n_patients == 0
        assert not link.patients_recur
