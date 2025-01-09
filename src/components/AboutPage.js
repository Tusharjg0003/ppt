// File: src/components/AboutPage.js
import React from 'react';
import './AboutPage.css';

const AboutPage = () => {
  return (
    <div className="about-container">
      <header className="about-header">
        <h1>About Us</h1>
      </header>
      <div className="about-content">
        <p>
          Established 15 years ago in Saudi Arabia, Production of Precision Trading Est. has been a trusted name in the industrial and construction sectors. We specialize in providing a wide range of high-quality products, including pipe support items, anchor bolts, lifting equipment, precast accessories, safety tools, and more.
        </p>
        <p>
          Our commitment to excellence and reliability has made us a preferred partner for contractors and businesses seeking durable and efficient solutions for their projects.
        </p>
      </div>
    </div>
  );
};

export default AboutPage;
