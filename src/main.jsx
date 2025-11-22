import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.jsx';
import { runStartupChecks } from './lib/startupChecks.js';
import './styles.css';

runStartupChecks();

const root = createRoot(document.getElementById('root'));
root.render(<App />);
