

### Setting Up Your Environment

1. Create a new file named **`.env`** in the root directory of your project.

2. Add the following line to the `.env` file:

   ```plaintext
   CLAUDE_API_KEY="[your api key]"
   ```

3. Save the file. 

Make sure to replace `[your api key]` with your actual Claude API key.

### Viewing Results

The repository includes a simple web viewer (`experiment-viewer/index.html`) for exploring the experimental results. To use it:

1. Start a local server in the project directory:
```bash
python -m http.server 8000
```

1. Open `http://localhost:8000/` in your browser

The viewer allows you to:
- Browse all result files in chronological order
- See accuracy statistics for both public and secret tasks
- Filter results to show only successful cases
- View detailed responses and predictions

No additional dependencies required - the viewer uses a single HTML file with CDN-loaded Tailwind CSS.