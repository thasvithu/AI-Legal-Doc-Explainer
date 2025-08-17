# Deploy to Hugging Face Spaces

1. Create a new Space (Streamlit) named `ai-legal-doc-explainer`.
2. Select SDK: **Streamlit**, hardware: CPU Basic.
3. Add secrets in Space Settings > Variables:
   - `GOOGLE_API_KEY` (optional)
   - `HF_TOKEN` (optional for inference API)
4. Upload repository files or connect Git.
5. Ensure `requirements.txt` is present. Spaces auto-installs.
6. (Optional) Add `runtime.txt` if Python version needs pin.
7. App starts with `streamlit run app.py` automatically.

On startup, if Gemini key is absent, app falls back to local HF model (may be slower on CPU).
