"""Streamlit-based dashboard for HRAG-Bot.

Launched via ``hrag gui`` (see cli.py::cmd_gui). The CLI shells out to
``streamlit run`` on this package's ``app.py`` because Streamlit's runtime
has to own the process. Heavy GUI deps (streamlit, pandas) live in the
``gui`` optional-dependency group so the rest of the package stays light.
"""
