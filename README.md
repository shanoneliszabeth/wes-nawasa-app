# Wes Nawasa App

Streamlit app deployed from this repository.

How to run locally:

```bash
pip install -r requirements.txt
streamlit run wes_nawasa_app.py
```

Deployment (Streamlit Cloud):

1. Push this repo to GitHub.
2. On https://share.streamlit.io, sign in with GitHub and create a new app pointing to this repo and `wes_nawasa_app.py`.

No Secrets configuration is needed — each visitor enters their own Gemini API key in the app's sidebar, so there is no shared `GOOGLE_API_KEY` to set.
