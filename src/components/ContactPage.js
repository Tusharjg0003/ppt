// File: src/components/ContactPage.js
import React from 'react';
import './ContactPage.css';

const ContactPage = () => {
  return (
    <div className="contact-container">
      <header className="contact-header">
        <h1>Contact Us</h1>
      </header>
      <div className="map-container">
        <iframe
          src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d74640.12960252613!2d50.07879255166017!3d26.420682681657872!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x3e49ef2b7e1c66f9%3A0x547674e49b8497b8!2sBilal%20Ibn%20Rabah%20St%2C%20Madinatul%20ummal%2C%20Dammam%2032253%2C%20Saudi%20Arabia!5e0!3m2!1sen!2s!4v1673621019477!5m2!1sen!2s"
          allowFullScreen=""
          loading="lazy"
          title="Google Maps Location"
        ></iframe>
      </div>
      <div className="contact-box">
        <div className="section left">
          <h2>Inquiries</h2>
          <p>
            For any inquiries, questions, or commendations, please contact us:
            <br />
            <strong>Phone:</strong> +966 538792814
            <br />
            <strong>Email:</strong>{' '}
            <a href="mailto:sajid_155@rediffmail.com">sajid_155@rediffmail.com</a>
          </p>
        </div>
        <div className="section right">
          <h2>Head Office</h2>
          <p>
            <strong>Address:</strong>
            <br />
            Bilal Ibn Rabah St, Madinatul ummal, Dammam 32253, Saudi Arabia
          </p>
        </div>
      </div>
    </div>
  );
};

export default ContactPage;
