# MongoDB Setup for ClauseInsight

## 📦 What's Stored

The system now logs all user queries and responses to MongoDB:

- User query text
- System response (clause, explanation, confidence)
- Query type (query/analysis)
- Timestamp
- Metadata (relevance score, confidence level)

## 🚀 Quick Setup

### Option 1: Local MongoDB (Recommended for Development)

1. **Install MongoDB:**

   - **Windows:** Download from https://www.mongodb.com/try/download/community
   - **Mac:** `brew install mongodb-community`
   - **Linux:** `sudo apt-get install mongodb`

2. **Start MongoDB:**

   ```bash
   # Windows
   mongod --dbpath C:\data\db

   # Mac/Linux
   mongod --dbpath /usr/local/var/mongodb
   ```

3. **Install Python dependency:**

   ```bash
   pip install motor==3.3.2
   ```

4. **Configure .env:**
   ```env
   MONGODB_URL=mongodb://localhost:27017
   ```

### Option 2: MongoDB Atlas (Cloud - Free Tier Available)

1. Create account at https://www.mongodb.com/cloud/atlas
2. Create a free cluster
3. Get connection string
4. Update `.env`:
   ```env
   MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority
   ```

## 📊 Database Structure

**Database:** `clause_insight`  
**Collection:** `query_history`

### Document Schema:

```json
{
  "user_id": "default_user",
  "query": "What is the termination clause?",
  "response": {
    "clause": {...},
    "explanation": {...},
    "relevance": {...}
  },
  "query_type": "query",
  "timestamp": "2026-01-16T10:30:00Z",
  "metadata": {
    "clause_title": "Termination Provisions",
    "confidence": 90,
    "relevance_score": 0.85
  }
}
```

## 🔍 Querying Data

### View Recent Queries:

```javascript
// In MongoDB shell or Compass
db.query_history.find().sort({ timestamp: -1 }).limit(10);
```

### Count Queries:

```javascript
db.query_history.countDocuments();
```

### Find High-Confidence Responses:

```javascript
db.query_history.find({ "metadata.confidence": { $gte: 85 } });
```

## ⚙️ No MongoDB? No Problem!

The system gracefully handles MongoDB connection failures:

- If MongoDB is not available, queries still work normally
- Logging is skipped with a warning message
- No impact on core RAG functionality

## 🔧 Troubleshooting

**Connection Refused:**

```bash
# Make sure MongoDB is running
mongod --version
# Check if process is running
ps aux | grep mongod  # Mac/Linux
tasklist | findstr mongod  # Windows
```

**Authentication Error:**

- Check username/password in connection string
- Ensure user has read/write permissions

**Python Error:**

```bash
# Reinstall motor
pip uninstall motor
pip install motor==3.3.2
```
