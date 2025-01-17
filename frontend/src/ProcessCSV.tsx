import React, { useState, ChangeEvent, FormEvent } from "react";

interface ApiResponse {
  status: string;
  data: any; // Adjust the type based on the actual response structure
}

const CsvUploader: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [response, setResponse] = useState<ApiResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string>("");

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setResponse(null);
      setError("");
    }
  };

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    if (!file) {
      setError("Please upload a CSV file.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setIsLoading(true);
    setError("");
    setResponse(null);

    try {
      const res = await fetch("http://127.0.0.1:8000/process_file/", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || "Error processing the file.");
      }

      const data: ApiResponse = await res.json();
      setResponse(data);
    } catch (err: any) {
      setError(err.message || "An unknown error occurred.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleAskLLM = async () => {
    const res = await fetch(`http://127.0.0.1:8000/ask-llm`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });

    if (res.ok) {
      const data = await res.json();
      console.log(data)
    } else {
    }
  }

  return (
    <div style={{ padding: "20px", maxWidth: "600px", margin: "auto" }}>
      <h2>Upload CSV File</h2>
      <form onSubmit={handleSubmit} style={{ marginBottom: "20px" }}>
        <div style={{ marginBottom: "10px" }}>
          <input
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            style={{ padding: "10px" }}
          />
        </div>
        <button
          type="submit"
          style={{ padding: "10px 20px" }}
          disabled={isLoading}
        >
          {isLoading ? "Processing..." : "Upload"}
        </button>
       
      </form>

      {error && <p style={{ color: "red" }}>{error}</p>}
      {response && (
        <div>
          <h3>Processing Result:</h3>
          <pre style={{ whiteSpace: "pre-wrap", padding: "10px" }}>
            {JSON.stringify(response, null, 2)}
          </pre>
        </div>
      )}
       <button
        onClick={handleAskLLM}
          style={{ padding: "10px 20px" }}
          disabled={isLoading}
        >
          {isLoading ? "Processing..." : "Ask LLM"}
        </button>
    </div>
    
  );
};

export default CsvUploader;
