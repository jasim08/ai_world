import { useState } from "react";

export default function App() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [file, setFile] = useState(null);

  const handleAsk = async () => {
    setLoading(true);
    setError("");
    setResponse("");

    try {
      const res = await fetch("http://localhost:8001/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      const data = await res.json();
      if (res.ok) {
        setResponse(data.answer || "No response");
      } else {
        setError(data.error || "Something went wrong");
      }
    } catch (err) {
      setError("Failed to connect to backend");
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please choose a file first");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch("http://localhost:8001/upload", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (res.ok) {
        setResponse(`File uploaded and embedded: ${data.status}`);
      } else {
        setError(data.error || "Upload failed");
      }
    } catch (err) {
      setError("Failed to upload file");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 p-10 font-sans">
      <div className="max-w-3xl mx-auto bg-white shadow-xl rounded-xl p-8">
        <h1 className="text-3xl font-bold text-center mb-6">RAG + vLLM Demo UI</h1>

        {/* Question Input */}
        <textarea
          className="w-full border p-3 rounded-md mb-4"
          rows="4"
          placeholder="Ask me anything from your documents..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />

        <button
          onClick={handleAsk}
          className="w-full bg-blue-600 text-white py-2 rounded-md hover:bg-blue-700"
          disabled={loading}
        >
          {loading ? "Processing..." : "Ask"}
        </button>

        {/* Upload Section */}
        <div className="mt-8">
          <h2 className="font-semibold mb-2">Upload PDF / Image / Document</h2>

          <input
            type="file"
            className="w-full border p-2 rounded-md mb-3"
            onChange={(e) => setFile(e.target.files[0])}
          />

          <button
            onClick={handleUpload}
            className="w-full bg-green-600 text-white py-2 rounded-md hover:bg-green-700"
            disabled={loading}
          >
            Upload + Embed
          </button>
        </div>

        {/* Error */}
        {error && (
          <div className="mt-4 p-3 bg-red-200 text-red-800 rounded-md">
            ❌ {error}
          </div>
        )}

        {/* Response */}
        {response && (
          <div className="mt-6 p-4 bg-gray-200 rounded-md whitespace-pre-wrap">
            <h3 className="font-bold mb-2">Response:</h3>
            {response}
          </div>
        )}
      </div>
    </div>
  );
}
