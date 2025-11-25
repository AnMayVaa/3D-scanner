const { app, BrowserWindow } = require('electron');
const path = require('path');

// Require the server script to start Node.js backend immediately
require('./server.js'); 

function createWindow() {
    // Create the browser window
    const win = new BrowserWindow({
        width: 1024,
        height: 800,
        title: "3D Scanner Pro",
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false
        }
    });

    // Load the localhost URL served by server.js
    // We use a small timeout to ensure the server is ready
    setTimeout(() => {
        win.loadURL('http://localhost:3000');
    }, 1000);
}

// App Ready Event
app.whenReady().then(() => {
    createWindow();

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        }
    });
});

// Quit when all windows are closed
app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});