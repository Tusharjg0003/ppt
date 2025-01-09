// File: src/components/ServicesPage.js
import React from 'react';
import './ServicesPage.css';

const ServicesPage = () => {
  return (
    <div className="services-container">
      <header className="services-header">
        <h1>What Products Do We Provide?</h1>
      </header>
      <div className="services-list">
        <div className="service-item">
          <h2>Pipe Supports</h2>
          <p>We offer top-notch pipe supports designed to meet the specific needs of your construction projects. Our expertise ensures reliable and durable solutions.</p>
        </div>
        <div className="service-item">
          <h2>Anchor Bolts</h2>
          <p>From standard to custom anchor bolts, we provide high-quality products that offer stability and safety for various structures and equipment.</p>
        </div>
        <div className="service-item">
          <h2>Lifting Equipment</h2>
          <p>Our lifting equipment range includes a variety of solutions to facilitate safe and efficient lifting operations in construction and industrial settings.</p>
        </div>
        <div className="service-item">
          <h2>Safety Equipment</h2>
          <p>We offer a comprehensive range of safety equipment, including helmets, harnesses, gloves, and protective gear, ensuring that your team operates in a secure environment while meeting safety regulations.</p>
        </div>
        <div className="service-item">
          <h2>Construction Tools</h2>
          <p>Our inventory includes a variety of construction tools, from power drills and saws to hand tools, catering to the diverse needs of on-site construction projects.</p>
        </div>
        <div className="service-item">
          <h2>Scaffolding Systems</h2>
          <p>We provide high-quality scaffolding systems designed for stability and ease of assembly, ensuring safe access to elevated work areas for construction workers.</p>
        </div>
      </div>
    </div>
  );
};

export default ServicesPage;
