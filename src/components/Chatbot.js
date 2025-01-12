import React, { useState } from "react";
import "./Chatbot.css";

const Chatbot = ({ onClose }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSendMessage = async () => {
    if (input.trim()) {
      const userMessage = { text: input, user: true };
      setMessages((prev) => [...prev, userMessage]);
      setInput("");
      setLoading(true);

      try {
        const response = await fetch("https://ppt-chatbot.onrender.com/ask", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ question: input }),
        });

        if (!response.ok) throw new Error("Server error");

        const data = await response.json();
        const botMessage = { text: data.answer, user: false };
        setMessages((prev) => [...prev, botMessage]);
      } catch (error) {
        console.error("Error:", error);
        const errorMessage = {
          text: "Something went wrong. Please try again later.",
          user: false,
        };
        setMessages((prev) => [...prev, errorMessage]);
      } finally {
        setLoading(false);
      }
    }
  };

  return (
    <div className="chatbot-container">
      <div className="chatbot-header">
        <span>Chat Assistant</span>
        <button className="chatbot-close-button" onClick={onClose}>
          ✖
        </button>
      </div>
      <div className="chatbot-body">
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`chatbot-message ${
              msg.user ? "chatbot-message-user" : "chatbot-message-bot"
            }`}
          >
            {msg.text}
          </div>
        ))}
        {loading && <div className="chatbot-typing-indicator">Typing...</div>}
      </div>
      <div className="chatbot-footer">
        <input
          type="text"
          placeholder="Type your message..."
          className="chatbot-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSendMessage()}
        />
        <button className="chatbot-send-button" onClick={handleSendMessage}>
          Send
        </button>
      </div>
    </div>
  );
};

export default Chatbot;
