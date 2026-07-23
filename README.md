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
3. In app Settings → Secrets add your Gemini (Google) API key as `GOOGLE_API_KEY`.
