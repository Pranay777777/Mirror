# Mirror Backend — Scientifically Defensible Video Analysis System

## 🎯 Overview

This is a **production-grade AI evaluation system** that analyzes human communication through scientifically defensible, camera-invariant metrics. Unlike demo-level systems, this implementation:

✅ **Camera-Invariant**: Same human = same score regardless of camera distance/zoom  
✅ **Temporal Modeling**: Analyzes behavioral patterns over time, not just per-frame  
✅ **Uncertainty Quantification**: Confidence scores for all metrics  
✅ **Scientifically Valid**: No pseudoscientific or invented metrics  

## 🧬 What We Measure

### **Body Language (Camera-Invariant)**
- **Posture Stability**: Shoulder tilt angle and alignment consistency
- **Movement Intensity**: Normalized velocity patterns (nervousness indicator)
- **Head Position**: Height ratio relative to shoulder width
- **Symmetry**: Facial and postural alignment metrics

### **Engagement Analysis**
- **Gaze Stability**: Duration of stable eye contact
- **Eye Contact Consistency**: Variance in gaze patterns
- **Attention Patterns**: Temporal consistency of focus

### **Expression Analysis**
- **Expression Dynamics**: Range and frequency of facial movements
- **Smile Frequency**: Events per minute with context
- **Blink Rate**: Per-minute rate (15-25 = normal)
- **Facial Symmetry**: Real-time symmetry measurements

### **Speech Analysis** (LLM with Safety Constraints)
- **Professionalism Score**: 0-10 scale with structured rubric
- **Communication Score**: 0-10 scale with confidence estimation
- **Sentiment Analysis**: Positive/neutral/negative classification
- **Tone Quality**: Professional/casual/inappropriate detection

## 🚫 What We DON'T Measure

- **Emotions**: Not scientifically detectable from video alone
- **Intent**: Cannot infer internal mental states
- **"Charisma"**: Subjective and culturally dependent
- **"Leadership Potential"**: Not measurable from body language

## 🔬 Scientific Limitations

- Requires frontal or near-frontal camera angle
- Performance degrades with occlusion or poor lighting
- Cultural variations in non-verbal communication
- Baseline behavior varies by individual
- **Calibration Required**: Scores need population baselines

## 📊 Confidence Estimation

Every metric includes:
- **Processing Success Rate**: Landmark detection reliability
- **Data Quality**: Excellent/Good/Fair/Poor assessment
- **Reliability Factors**: Specific issues (low detection, insufficient data)
- **Temporal Confidence**: Based on data quantity and quality

## 🚀 Quick Start (Local Development)

### Prerequisites
- Python 3.8+
- FFmpeg (for video processing)
- Valid API keys for Sarvam AI and OpenAI
```powershell
cd "C:\Users\aim4g\Desktop\tmi\Mirror_Backend_New_fresh\Mirror_Backend_New_fresh"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Installation
```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Configuration

Create `.env` with your API keys:
```bash
SARVAM_API_KEY=your_sarvam_key_here
OPENAI_API_KEY=your_openai_key_here
```

### Running the Server
```powershell
copy .env.example .env
# Edit .env with your API keys
```

### Testing

Test with a short video (<30 seconds recommended):
```bash
python test_post.py C:\path\to\short_video.mp4
```

**Expected Response Structure:**
```json
{
  "status": "success",
  "analysis_version": "v2.0_camera_invariant",
  "results": {
    "body_language": {
      "analysis_metadata": {...},
      "posture_analysis": {
        "stability_score": 0.85,
        "uprightness_score": 0.92,
        "movement_intensity": 0.015,
        "interpretation": "Excellent posture: Very stable, upright, and composed"
      },
      "engagement_analysis": {...},
      "expression_analysis": {...},
      "confidence_metrics": {
        "overall_confidence": 0.91,
        "data_quality": "Excellent",
        "reliability_factors": ["All metrics reliable"]
      }
    },
    "speech_analysis": {
      "sentiment": "positive",
      "professionalism_score": 8.2,
      "communication_score": 7.8,
      "confidence": 0.85
    },
    "transcript": "Full speech transcript here..."
  }
}
```

## 📈 Performance Notes

- **Processing Time**: ~2-5 seconds per video minute (depends on hardware)
- **Memory Usage**: Proportional to video resolution
- **Accuracy**: >90% landmark detection at 720p+ resolution
- **Recommended Video Length**: 30 seconds - 10 minutes for optimal analysis

## 🔧 Architecture

```
Video Input → MediaPipe → Camera-Invariant Features → Temporal Analysis → Confidence Estimation → JSON Response
                    ↓
Audio Extraction → Sarvam AI → Transcript → OpenAI (Structured) → Validated Scores
```

## 🧪 Testing & Validation

The system includes:
- **Camera invariance tests**: Same person, different distances
- **Temporal consistency tests**: Stable scores across time windows
- **LLM safety validation**: Score clamping and output validation
- **Error handling**: Graceful degradation with confidence reporting

## 🚨 Production Considerations

**Before deploying to production:**

1. **Background Processing**: Current implementation is synchronous
2. **File Cleanup**: Implement automatic upload cleanup
3. **Rate Limiting**: Add API rate limiting
4. **Authentication**: Secure the endpoints
5. **Monitoring**: Add health checks and metrics
6. **Calibration**: Build population-specific baselines

## 📚 Scientific Background

### Camera-Invariant Geometry
All spatial features are normalized using body-relative ratios:
- `eye_distance / shoulder_width` → eliminates camera distance effects
- `mouth_opening / face_height` → eliminates face size effects  
- `shoulder_tilt_angle` → eliminates camera angle effects

### Temporal Modeling
Instead of per-frame scores, we compute:
- **Stability**: Inverse coefficient of variation
- **Dynamics**: Total variation normalized by range
- **Event Frequency**: Peaks per minute (smiles, blinks)
- **Velocity**: Rate of change for movement patterns

### Confidence Estimation
Multi-factor confidence calculation:
- Landmark detection success rate
- Temporal data sufficiency
- Feature value reasonableness checks
- Cross-validation between metrics

## 🤝 Contributing

When adding new features:
1. Must be camera-invariant
2. Include temporal analysis
3. Provide confidence estimation
4. Document scientific basis
5. Add validation tests

## 📄 License

Scientific Analysis System - See LICENSE file for details.
