"""Report-text tooling: sampling, normalisation, and LLM-driven reconnaissance.

Training-time only. Nothing here may be imported by the inference path -- reports
exist for train studies and not for test ones, so anything that reads a report
cannot run at submission time.
"""
