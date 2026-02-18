# 🧪 COMPLETE TESTING GUIDE & MANUAL VERIFICATION

## 🏁 **QUICK REBOOT / RESTART GUIDE**

If you restart your laptop or close the terminals, follow these exact steps to restore the system:

### **1. Open Terminal (PowerShell or Command Prompt)**
Run `cmd` or `powershell` as Administrator (optional but recommended for path access).

### **2. Navigate to Project Directory**
```bash
cd C:\Users\aim4g\Desktop\tmi\Mirror_Backend_New_fresh\Mirror_Backend_New_fresh
```

### **3. Activate Virtual Environment**
This ensures you use the correct Python dependencies.
```bash
# PowerShell:
.\.venv\Scripts\Activate.ps1

# Command Prompt (cmd):
.venv\Scripts\activate.bat
```
*(You should see `(.venv)` appear at the start of your command prompt)*

### **4. Start the Server**
```bash
python main.py
```
*(Keep this terminal open! It runs the backend server at `http://127.0.0.1:8000`)*

### **5. Run Tests (Open a NEW Terminal)**
1. Open a **new** terminal window.
2. Navigate to project folder again (Step 2).
3. Activate environment (Step 3).
4. Run the analysis script:
   ```bash
   python test_post.py "C:\path\to\your\test_video.mp4"
   ```
   *(e.g., `python test_post.py "uploads\pran.mp4"`)*

---

## 🚀 **SYSTEM STATUS: ✅ RUNNING**

The scientifically defensible video analysis system is now running on:
```
http://127.0.0.1:8000
```

---

## 📋 **TESTING CHECKLIST**

### ✅ **AUTOMATED TESTS PASSED**
- [x] LLM Safety Validation (score clamping, range enforcement)
- [x] Temporal Feature Calculation (stability, dynamics)
- [x] Core Component Imports
- [x] Camera-Invariant Geometry Module
- [x] Scientific Scoring Logic

---

## 🎯 **MANUAL TESTING PROCEDURES**

### **1. API HEALTH CHECK**
```bash
curl http://127.0.0.1:8000/docs
```
**Expected**: FastAPI documentation page loads

### **2. VIDEO ANALYSIS TEST**

#### **Requirements:**
- Video file (MP4, MOV, etc.)
- <30 seconds recommended for testing
- Person facing camera (frontal or near-frontal)

#### **Test Command:**
```bash
python test_post.py "C:\path\to\your\test_video.mp4"
```

#### **Expected Response Structure:**
```json
{
  "status": "success",
  "analysis_version": "v2.0_camera_invariant",
  "results": {
    "body_language": {
      "analysis_metadata": {
        "total_frames": 450,
        "successful_frames": 423,
        "processing_success_rate": 0.94,
        "fps": 30.0,
        "temporal_window_seconds": 2.0
      },
      "posture_analysis": {
        "stability_score": 0.85,
        "uprightness_score": 0.92,
        "movement_intensity": 0.015,
        "interpretation": "Excellent posture: Very stable, upright, and composed"
      },
      "engagement_analysis": {
        "gaze_stability": 0.78,
        "eye_contact_consistency": 0.82,
        "stable_gaze_duration": 1.2,
        "interpretation": "High engagement: Very consistent eye contact and focused attention"
      },
      "expression_analysis": {
        "expression_dynamics": 0.65,
        "smile_frequency": 2.3,
        "blink_rate": 18.5,
        "interpretation": "Moderately expressive: Good facial expression variety"
      },
      "confidence_metrics": {
        "overall_confidence": 0.91,
        "data_quality": "Excellent",
        "reliability_factors": ["All metrics reliable"]
      }
    },
    "speech_analysis": {
      "sentiment": "positive",
      "tone_quality": "professional",
      "professionalism_score": 8.2,
      "communication_score": 7.8,
      "confidence": 0.85,
      "voice_quality_feedback": "Clear and well-paced delivery",
      "analysis_notes": "Strong communication skills demonstrated"
    },
    "transcript": "Full speech transcript appears here..."
  },
  "metadata": {
    "processing_method": "camera_invariant_geometry",
    "temporal_analysis": true,
    "confidence_estimation": true
  }
}
```

---

## 🔬 **SCIENTIFIC VALIDATION TESTS**

### **Test 1: Camera Invariance**
1. Record same person at different distances
2. Analyze both videos
3. **Expected**: Similar scores despite camera distance

### **Test 2: Temporal Consistency**
1. Analyze same video multiple times
2. **Expected**: Identical results (deterministic)

### **Test 3: Confidence Estimation**
1. Use poor quality video (low light, occlusion)
2. **Expected**: Lower confidence scores, appropriate reliability factors

### **Test 4: LLM Safety**
1. Test with very short or unclear transcript
2. **Expected**: Confidence < 0.5, appropriate default scores

---

## 📊 **SUCCESS METRICS**

### **Processing Success Rate**
- **Excellent**: >90% frames processed
- **Good**: 70-90% frames processed  
- **Fair**: 50-70% frames processed
- **Poor**: <50% frames processed

### **Confidence Scores**
- **High**: >0.8 confidence
- **Medium**: 0.5-0.8 confidence
- **Low**: <0.5 confidence

### **Score Ranges**
- **Posture/Engagement/Expression**: 0.0-1.0 (normalized)
- **Professionalism/Communication**: 0.0-10.0
- **LLM Confidence**: 0.0-1.0

---

## 🚨 **TROUBLESHOOTING**

### **Common Issues & Solutions**

#### **Issue: "insufficient_landmarks" Error**
**Cause**: Person not visible or poor video quality
**Solution**: Use better lighting, frontal angle, higher resolution

#### **Issue: "insufficient_temporal_data"**
**Cause**: Video too short (<3 seconds)
**Solution**: Use longer video (minimum 5 seconds recommended)

#### **Issue: Low Processing Success Rate**
**Cause**: Occlusion, fast movement, poor lighting
**Solution**: Improve video conditions, check camera positioning

#### **Issue: API Timeouts**
**Cause**: Long video or slow processing
**Solution**: Use shorter videos (<30 seconds for testing)

---

## 🧪 **ADVANCED TESTING**

### **Stress Testing**
```bash
# Test multiple concurrent requests
for i in {1..5}; do
  python test_post.py "test_video.mp4" &
done
wait
```

### **Error Handling Test**
```bash
# Test with invalid file
python test_post.py "nonexistent.mp4"
# Expected: Graceful error response
```

### **Memory Usage Test**
```bash
# Monitor memory during processing
# Windows: Use Task Manager
# Look for: Memory leaks, excessive usage
```

---

## 📈 **PERFORMANCE BENCHMARKS**

### **Expected Performance**
- **720p Video**: ~2-3 seconds processing time per minute
- **1080p Video**: ~3-5 seconds processing time per minute
- **Memory Usage**: ~100-200MB for typical videos
- **CPU Usage**: ~50-80% during processing

### **Optimization Tips**
- Use shorter videos for testing
- Ensure adequate RAM (4GB+ recommended)
- Close other applications during testing

---

## 🎯 **PRODUCTION READINESS CHECKLIST**

### **✅ Completed**
- [x] Camera-invariant features
- [x] Temporal analysis
- [x] Uncertainty quantification
- [x] LLM safety constraints
- [x] Scientific documentation
- [x] Error handling
- [x] Structured responses

### **⚠️ Before Production Deployment**
- [ ] Add authentication/authorization
- [ ] Implement file cleanup
- [ ] Add rate limiting
- [ ] Set up monitoring/logging
- [ ] Configure HTTPS
- [ ] Add health checks

---

## 📝 **TESTING LOG**

Keep track of your test results:

```
Date: ___________
Video File: ___________
Duration: ___________
Resolution: ___________

Results:
- Processing Success Rate: ____%
- Overall Confidence: ____
- Posture Score: ____
- Engagement Score: ____
- Expression Score: ____
- Speech Score: ____

Issues Found: ___________
```

---

## 🎉 **SUCCESS INDICATORS**

Your system is working correctly when:

1. ✅ **API responds without errors**
2. ✅ **Response structure matches expected format**
3. ✅ **All scores are within valid ranges**
4. ✅ **Confidence estimation is reasonable**
5. ✅ **Interpretations make sense for the video**
6. ✅ **Same video produces consistent results**

---

## 🆘 **GETTING HELP**

If issues persist:

1. **Check server logs**: Look for error messages
2. **Verify API keys**: Ensure .env file has valid keys
3. **Test with different videos**: Rule out video-specific issues
4. **Check system resources**: Ensure adequate CPU/RAM

---

**🚀 Your scientifically defensible video analysis system is ready for testing!**
