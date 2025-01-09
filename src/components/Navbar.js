// File: src/components/Navbar.js
import React from 'react';
import { Link } from 'react-router-dom';
import './Navbar.css';
import logo from '../images/PPT-logo.png'; // Adjust the path to your logo

const Navbar = () => {
  return (
    <nav className="navbar">
      <div className="navbar-logo">
        <img src={logo} alt="Company Logo" />
        <span>Production of Precision Trading Est.</span>
      </div>
      <ul className="navbar-links">
        <li><Link to="/">Home</Link></li>
        <li><Link to="/services">Products</Link></li>
        <li><Link to="/about">About</Link></li>
        <li><Link to="/contact">Contact</Link></li>
      </ul>
    </nav>
  );
};

export default Navbar;
