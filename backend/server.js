const express = require('express');
const cors = require('cors');
const bcrypt = require('bcryptjs');
const db = require('./database');

const app = express();
const PORT = 3001;

app.use(cors({ origin: 'http://localhost:5173' }));
app.use(express.json());

app.post('/api/signup', async (req, res) => {
  const { username, email, password } = req.body;

  if (!username || !email || !password) {
    return res.status(400).json({ error: 'All fields are required.' });
  }

  const users = db.get('users');
  const existing = users.find(u => u.username === username || u.email === email).value();
  if (existing) {
    return res.status(409).json({ error: 'Username or email already exists.' });
  }

  const hashed = await bcrypt.hash(password, 10);
  const newUser = { id: Date.now(), username, email, password: hashed, createdAt: new Date().toISOString() };
  users.push(newUser).write();

  res.status(201).json({ message: 'Account created successfully.' });
});

app.post('/api/login', async (req, res) => {
  const { username, password } = req.body;

  if (!username || !password) {
    return res.status(400).json({ error: 'All fields are required.' });
  }

  const user = db.get('users').find({ username }).value();
  if (!user) {
    return res.status(401).json({ error: 'Invalid username or password.' });
  }

  const match = await bcrypt.compare(password, user.password);
  if (!match) {
    return res.status(401).json({ error: 'Invalid username or password.' });
  }

  res.json({ message: 'Login successful.', user: { id: user.id, username: user.username, email: user.email } });
});

app.listen(PORT, () => {
  console.log(`Backend running at http://localhost:${PORT}`);
});
