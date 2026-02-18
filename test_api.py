import requests
import json

def test_api_endpoint():
    """Test the analyze endpoint with a simple health check."""
    
    print("🧪 TESTING API ENDPOINT")
    print("=" * 40)
    
    # Test 1: Check API is accessible
    try:
        response = requests.get("http://127.0.0.1:8000/docs", timeout=5)
        if response.status_code == 200:
            print("✅ API documentation accessible")
        else:
            print(f"❌ API returned status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API not accessible: {e}")
        return False
    
    # Test 2: Test endpoint exists (will fail without file, but should return proper error)
    try:
        response = requests.post("http://127.0.0.1:8000/analyze", timeout=5)
        if response.status_code == 422:  # Validation error (expected)
            print("✅ /analyze endpoint exists and validates input")
        else:
            print(f"⚠️  Unexpected status code: {response.status_code}")
    except Exception as e:
        print(f"❌ Endpoint test failed: {e}")
        return False
    
    print("\n🎉 API IS READY FOR VIDEO TESTING!")
    print("\n📝 Next Steps:")
    print("1. Get a test video file")
    print("2. Run: python test_post.py path/to/video.mp4")
    print("3. Check response structure matches expected format")
    
    return True

if __name__ == "__main__":
    test_api_endpoint()
