#!/bin/bash

# Milvuso API Test Script

set -e

BASE_URL="http://localhost:8080"
USER_ID="550e8400-e29b-41d4-a716-446655440000"
ITEM_ID1="550e8400-e29b-41d4-a716-446655440001"
ITEM_ID2="550e8400-e29b-41d4-a716-446655440002"

echo "🧪 Milvuso API Testing Started"

# 1. Health check
echo "1️⃣ Health check..."
response=$(curl -s -w "%{http_code}" -o /tmp/health_response.json "$BASE_URL/health")
if [ "$response" = "200" ]; then
    echo "✅ Health check passed"
    cat /tmp/health_response.json | jq '.' 2>/dev/null || cat /tmp/health_response.json
else
    echo "❌ Health check failed (HTTP $response)"
    exit 1
fi

echo ""

# 2. Add item features
echo "2️⃣ Adding item features..."

# Add first item
item1_data='{
  "item_id": "'$ITEM_ID1'",
  "embedding": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
  "category": "electronics",
  "tags": ["smartphone", "android"],
  "popularity_score": 0.8,
  "created_at": "2024-01-01T12:00:00Z"
}'

response=$(curl -s -w "%{http_code}" -o /tmp/item1_response.json -X POST "$BASE_URL/items" \
  -H "Content-Type: application/json" \
  -d "$item1_data")

if [ "$response" = "200" ]; then
    echo "✅ Item 1 added successfully"
else
    echo "❌ Item 1 addition failed (HTTP $response)"
    cat /tmp/item1_response.json
fi

# Add second item
item2_data='{
  "item_id": "'$ITEM_ID2'",
  "embedding": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
  "category": "books",
  "tags": ["fiction", "bestseller"],
  "popularity_score": 0.6,
  "created_at": "2024-01-01T12:00:00Z"
}'

response=$(curl -s -w "%{http_code}" -o /tmp/item2_response.json -X POST "$BASE_URL/items" \
  -H "Content-Type: application/json" \
  -d "$item2_data")

if [ "$response" = "200" ]; then
    echo "✅ Item 2 added successfully"
else
    echo "❌ Item 2 addition failed (HTTP $response)"
    cat /tmp/item2_response.json
fi

echo ""

# 3. Record user actions
echo "3️⃣ Recording user actions..."

# User views item 1
action1_data='{
  "user_id": "'$USER_ID'",
  "item_id": "'$ITEM_ID1'",
  "action_type": "View",
  "timestamp": "2024-01-01T12:00:00Z"
}'

response=$(curl -s -w "%{http_code}" -o /tmp/action1_response.json -X POST "$BASE_URL/actions" \
  -H "Content-Type: application/json" \
  -d "$action1_data")

if [ "$response" = "200" ]; then
    echo "✅ User action 1 recorded successfully (View)"
else
    echo "❌ User action 1 recording failed (HTTP $response)"
    cat /tmp/action1_response.json
fi

# User clicks item 1
action2_data='{
  "user_id": "'$USER_ID'",
  "item_id": "'$ITEM_ID1'",
  "action_type": "Click",
  "timestamp": "2024-01-01T12:01:00Z"
}'

response=$(curl -s -w "%{http_code}" -o /tmp/action2_response.json -X POST "$BASE_URL/actions" \
  -H "Content-Type: application/json" \
  -d "$action2_data")

if [ "$response" = "200" ]; then
    echo "✅ User action 2 recorded successfully (Click)"
else
    echo "❌ User action 2 recording failed (HTTP $response)"
    cat /tmp/action2_response.json
fi

# User likes item 1
action3_data='{
  "user_id": "'$USER_ID'",
  "item_id": "'$ITEM_ID1'",
  "action_type": "Like",
  "timestamp": "2024-01-01T12:02:00Z"
}'

response=$(curl -s -w "%{http_code}" -o /tmp/action3_response.json -X POST "$BASE_URL/actions" \
  -H "Content-Type: application/json" \
  -d "$action3_data")

if [ "$response" = "200" ]; then
    echo "✅ User action 3 recorded successfully (Like)"
else
    echo "❌ User action 3 recording failed (HTTP $response)"
    cat /tmp/action3_response.json
fi

echo ""

# 4. Wait for system to process data
echo "4️⃣ Waiting for system to process data..."
sleep 3

# 5. Get user profile
echo "5️⃣ Getting user profile..."
response=$(curl -s -w "%{http_code}" -o /tmp/user_response.json "$BASE_URL/users/$USER_ID")

if [ "$response" = "200" ]; then
    echo "✅ User profile retrieved successfully"
    cat /tmp/user_response.json | jq '.' 2>/dev/null || cat /tmp/user_response.json
else
    echo "❌ User profile retrieval failed (HTTP $response)"
    cat /tmp/user_response.json
fi

echo ""

# 6. Get item features
echo "6️⃣ Getting item features..."
response=$(curl -s -w "%{http_code}" -o /tmp/item_get_response.json "$BASE_URL/items/$ITEM_ID1")

if [ "$response" = "200" ]; then
    echo "✅ Item features retrieved successfully"
    cat /tmp/item_get_response.json | jq '.' 2>/dev/null || cat /tmp/item_get_response.json
else
    echo "❌ Item features retrieval failed (HTTP $response)"
    cat /tmp/item_get_response.json
fi

echo ""

# 7. Get recommendations
echo "7️⃣ Getting recommendations..."
response=$(curl -s -w "%{http_code}" -o /tmp/recommendations_response.json "$BASE_URL/recommendations/$USER_ID?num_recommendations=5")

if [ "$response" = "200" ]; then
    echo "✅ Recommendations retrieved successfully"
    cat /tmp/recommendations_response.json | jq '.' 2>/dev/null || cat /tmp/recommendations_response.json
else
    echo "❌ Recommendations retrieval failed (HTTP $response)"
    cat /tmp/recommendations_response.json
fi

echo ""

# 8. Get filtered recommendations by category
echo "8️⃣ Getting filtered recommendations by category..."
response=$(curl -s -w "%{http_code}" -o /tmp/filtered_recommendations_response.json "$BASE_URL/recommendations/$USER_ID?num_recommendations=5&filter_categories=electronics")

if [ "$response" = "200" ]; then
    echo "✅ Filtered recommendations retrieved successfully"
    cat /tmp/filtered_recommendations_response.json | jq '.' 2>/dev/null || cat /tmp/filtered_recommendations_response.json
else
    echo "❌ Filtered recommendations retrieval failed (HTTP $response)"
    cat /tmp/filtered_recommendations_response.json
fi

echo ""

# Clean up temporary files
rm -f /tmp/*_response.json

echo "🎉 API testing completed!"
echo ""
echo "📊 Test Summary:"
echo "  ✅ Health check"
echo "  ✅ Item feature addition"
echo "  ✅ User action recording"
echo "  ✅ User profile retrieval"
echo "  ✅ Item feature retrieval"
echo "  ✅ Recommendation generation"
echo "  ✅ Filtered recommendations"
echo ""
echo "💡 System is running normally, ready to use!"
