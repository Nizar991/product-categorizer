import React, { useState } from "react";
import "./App.css";

function App() {
  const [formData, setFormData] = useState({
    brand: "",
    title: "",
    description: "",
    short_desc: "",
    spec: "",
  });

  const [prediction, setPrediction] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    console.log("Submitting to FastAPI:", formData);
    setLoading(true);
    setError("");
    setPrediction("");

    try {
      const response = await fetch("http://127.0.0.1:8002/predict_category", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const data = await response.json();
      if (data.predicted_category) {
        setPrediction(data.predicted_category);
      } else {
        setError(data.error || "Prediction failed");
      }
    } catch (err) {
      setError(err.message);
    }

    setLoading(false);
  };

  return (
    <div className="App">
      
      <div className="form-card">
      <h1>Product Category Predictor</h1>
      <form onSubmit={handleSubmit}>
        <input
            name="brand"
            placeholder="Brand"
            value={formData.brand}
            onChange={handleChange}
          />
          <input
            name="title"
            placeholder="Title"
            value={formData.title}
            onChange={handleChange}
          />
          <input
            name="description"
            placeholder="Description"
            value={formData.description}
            onChange={handleChange}
          />
          <input
            name="short_desc"
            placeholder="Short Description"
            value={formData.short_desc}
            onChange={handleChange}
          />
          <input
            name="spec"
            placeholder="Specification"
            value={formData.spec}
            onChange={handleChange}
          />
          <button type="submit" disabled={loading}>
            {loading ? "Predicting..." : "Predict Category"}
          </button>
      </form>

      {prediction && <h2 className="result">Predicted Category: {prediction}</h2>}
      {error && <h2 className="error">{error}</h2>}
      </div>
    </div>
    
  );
}

export default App;
