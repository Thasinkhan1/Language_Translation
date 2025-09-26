async function translateText() {
  const sentence = document.getElementById("inputText").value;
  const outputBox = document.getElementById("outputText");

  outputBox.innerText = "⏳ Translating...";

  try {
    const response = await fetch("http://127.0.0.1:8000/translate/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sentence: sentence })
    });

    const data = await response.json();
    outputBox.innerText = "➡ " + (data.translation || "No translation found");
  } catch (error) {
    outputBox.innerText = "❌ Error: Unable to connect to server.";
  }
}
