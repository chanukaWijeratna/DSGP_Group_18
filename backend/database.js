const low = require('lowdb');
const FileSync = require('lowdb/adapters/FileSync');
const path = require('path');

const adapter = new FileSync(path.join(__dirname, 'users.json'));
const db = low(adapter);

db.defaults({ users: [] }).write();

module.exports = db;
