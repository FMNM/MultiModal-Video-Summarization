// Get elements
const homePage = document.getElementById("home-page");
const resultsPage = document.getElementById("results-page");
const videoUrlInput = document.getElementById("video-url");
const submitBtn = document.getElementById("submit-btn");

const videoContainer = document.getElementById("video-container");
const overallSummaryElem = document.getElementById("overall-summary");
const topicSummariesElem = document.getElementById("topic-summaries");

const chatMessagesElem = document.getElementById("chat-messages");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");

let sessionId = "";

// Event listener for submit button
submitBtn.addEventListener("click", async () => {
  const videoUrl = videoUrlInput.value.trim();
  if (!videoUrl) {
    alert("Please enter a video URL.");
    return;
  }

  // Send video URL to backend
  try {
    submitBtn.disabled = true;
    submitBtn.textContent = "Processing...";

    const response = await fetch("/process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_link: videoUrl }),
    });

    const data = await response.json();
    if (response.ok) {
      sessionId = data.session_id;
      displayResults(videoUrl, data);
    } else {
      alert(data.error || "An error occurred while processing the video.");
    }
  } catch (error) {
    alert("An error occurred: " + error.message);
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Submit";
  }
});

// Function to display results
function displayResults(videoUrl, data) {
  homePage.classList.add("hidden");
  resultsPage.classList.remove("hidden");

  // Embed video
  const videoId = extractYouTubeID(videoUrl);
  if (videoId) {
    videoContainer.innerHTML = `<iframe width="100%" height="400" src="https://www.youtube.com/embed/${videoId}" frameborder="0" allowfullscreen></iframe>`;
  } else {
    videoContainer.innerHTML = "<p>Unable to embed video.</p>";
  }

  // Display summaries
  overallSummaryElem.textContent = data.overall_summary;

  // Display topic summaries
  topicSummariesElem.innerHTML = "";
  for (const [topic, summaryData] of Object.entries(data.topic_summaries)) {
    const topicDiv = document.createElement("div");
    topicDiv.classList.add("topic-summary");

    const topicTitle = document.createElement("h3");
    topicTitle.textContent = `Topic ${topic} (Time: ${formatTime(
      summaryData.start_time
    )} - ${formatTime(summaryData.end_time)}):`;
    topicDiv.appendChild(topicTitle);

    const topicSummary = document.createElement("p");
    topicSummary.textContent = summaryData.summary;
    topicDiv.appendChild(topicSummary);

    topicSummariesElem.appendChild(topicDiv);
  }
}

// Event listener for send button in chatbox
sendBtn.addEventListener("click", async () => {
  const question = chatInput.value.trim();
  if (!question) {
    alert("Please enter a question.");
    return;
  }

  // Display user's question
  addChatMessage("You", question);

  chatInput.value = "";
  sendBtn.disabled = true;
  sendBtn.textContent = "Thinking...";

  // Send question to backend
  try {
    const response = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, session_id: sessionId }),
    });

    const data = await response.json();
    if (response.ok) {
      // Display assistant's answer
      addChatMessage("Assistant", data.answer);
    } else {
      addChatMessage(
        "Assistant",
        data.error || "An error occurred while processing your question."
      );
    }
  } catch (error) {
    addChatMessage("Assistant", "An error occurred: " + error.message);
  } finally {
    sendBtn.disabled = false;
    sendBtn.textContent = "Send";
  }
});

// Function to add chat messages
function addChatMessage(sender, message) {
  const messageDiv = document.createElement("div");
  messageDiv.classList.add("chat-message");
  messageDiv.classList.add(sender === "You" ? "user" : "assistant");
  messageDiv.textContent = `${sender}: ${message}`;
  chatMessagesElem.appendChild(messageDiv);
  chatMessagesElem.scrollTop = chatMessagesElem.scrollHeight;
}

// Function to extract YouTube video ID
function extractYouTubeID(url) {
  const regExp =
    /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|\&v=)([^#\&\?]*).*/;
  const match = url.match(regExp);
  if (match && match[2].length == 11) {
    return match[2];
  } else {
    return null;
  }
}

// Function to format time in seconds to HH:MM:SS
function formatTime(seconds) {
  const date = new Date(0);
  date.setSeconds(seconds);
  return date.toISOString().substr(11, 8);
}
