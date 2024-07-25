const express = require('express');
const app = express();
const port = 3005;

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Routes
app.get('/', (req, res) => {
  res.send('Hello World!');
});

app.get('/api/design/remove-background', require('./DesignController/removebg.js'));

// Start the server
app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
