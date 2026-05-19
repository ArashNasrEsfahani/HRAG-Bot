"""Custom UI components rendered via streamlit.components.v1.

Each module exposes a ``render(payload)`` (or ``render_tree(payload)``)
function that returns the HTML string + a recommended pixel height for
``components.v1.html(html, height=...)``. The iframe rendering bypasses
Streamlit's HTML sanitation and gives us full browser capabilities
(JavaScript, SVG, custom animations) that ``st.markdown`` does not.
"""
