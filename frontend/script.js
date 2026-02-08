document
  .getElementById("news-form")
  .addEventListener("submit", async function (e) {
    e.preventDefault();

    const textInput = document.getElementById("news-input").value.trim();
    const resultDiv = document.getElementById("result");
    const loadingDiv = document.getElementById("loading");
    const labelSpan = document.getElementById("prediction-label");
    const confidenceSpan = document.getElementById("confidence-score");
    const errorDiv = document.getElementById("error");

    // Clear previous results/errors
    errorDiv.classList.add("d-none");
    resultDiv.classList.add("d-none");

    if (!textInput) {
      errorDiv.textContent = "Please enter some text to check.";
      errorDiv.classList.remove("d-none");
      return;
    }

    // Show loading
    loadingDiv.classList.remove("d-none");

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: textInput }),
      });

      const data = await response.json();

      // Hide loading
      loadingDiv.classList.add("d-none");

      if (response.ok) {
        const prediction = data.prediction; // "Fake" or "Real"
        const confidence = (data.confidence * 100).toFixed(2);

        labelSpan.textContent = prediction;
        confidenceSpan.textContent = confidence;

        if (prediction === "Real") {
          labelSpan.className = "badge bg-success"; // Green for Real
        } else {
          labelSpan.className = "badge bg-danger"; // Red for Fake
        }

        resultDiv.classList.remove("d-none");
      } else {
        errorDiv.textContent =
          data.error || "An error occurred during prediction.";
        errorDiv.classList.remove("d-none");
      }
    } catch (error) {
      console.error("Error:", error);
      loadingDiv.classList.add("d-none");
      errorDiv.textContent =
        "Failed to communicate with the server. Please check your connection.";
      errorDiv.classList.remove("d-none");
    }
  });
