import React, { useState, Suspense } from "react";
import "./HomePage.css";

// Lazy load the Chatbot component
const Chatbot = React.lazy(() => import("./Chatbot"));

const HomePage = () => {
  const [isChatbotOpen, setIsChatbotOpen] = useState(false);

  const toggleChatbot = () => {
    setIsChatbotOpen((prev) => !prev);
  };

  return (
    <div className="homepage-container">
      <header className="homepage-header">
        <h1>SERVING SAUDI ARABIA FOR 15 YEARS</h1>
        <p>PROVIDING HIGH-QUALITY PRODUCTS AND EXCEPTIONAL SERVICE</p>
      </header>

      {/* Chat with Us Button */}
      {!isChatbotOpen && (
        <button className="chat-with-us-button" onClick={toggleChatbot}>
          Chat with Us
        </button>
      )}

      {/* Render Chatbot when opened */}
      {isChatbotOpen && (
        <Suspense fallback={null}>
          <Chatbot onClose={toggleChatbot} />
        </Suspense>
      )}
    </div>
  );
};

export default HomePage;
