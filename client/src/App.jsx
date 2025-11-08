import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import {
  Camera,
  Upload,
  X,
  AlertCircle,
  CheckCircle,
  Loader2,
  Video,
  Image as ImageIcon,
  ZoomIn,
  FileText,
  Car,
  Shield,
  Sparkles,
} from "lucide-react";

const API_BASE_URL = "http://localhost:8000";

function App() {
  const [cameraOn, setCameraOn] = useState(false);
  const [capturedImage, setCapturedImage] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [annotatedImageUrl, setAnnotatedImageUrl] = useState(null);
  const [apiStatus, setApiStatus] = useState(null);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const streamRef = useRef(null);

  useEffect(() => {
    checkApiHealth();
  }, []);

  const checkApiHealth = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/health`);
      setApiStatus(response.data);
    } catch (err) {
      console.error("API health check failed:", err);
      setApiStatus({ status: "offline", ready: false });
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: 1280, height: 720 },
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setCameraOn(true);
        setError(null);
      }
    } catch (err) {
      setError("Camera access denied. Please allow camera permissions.");
      console.error("Camera error:", err);
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraOn(false);
  };

  const captureFrame = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (video && canvas) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0);

      canvas.toBlob((blob) => {
        if (blob) {
          setCapturedImage(blob);
          setPreviewUrl(URL.createObjectURL(blob));
          setUploadedFile(null);
          setAnnotatedImageUrl(null);
          stopCamera();
        }
      }, "image/jpeg", 0.95);
    }
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (!file.type.startsWith("image/")) {
        setError("Please upload an image file");
        return;
      }
      setUploadedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setCapturedImage(null);
      setAnnotatedImageUrl(null);
      if (cameraOn) stopCamera();
      setError(null);
    }
  };

  const analyzeImage = async () => {
    const imageToAnalyze = capturedImage || uploadedFile;
    if (!imageToAnalyze) {
      setError("Please capture or upload an image first");
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);
    setAnnotatedImageUrl(null);

    try {
      const formData = new FormData();
      formData.append("file", imageToAnalyze, "plate.jpg");

      const response = await axios.post(`${API_BASE_URL}/detect`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setResults(response.data);
      
      if (response.data.annotated_image) {
        setAnnotatedImageUrl(response.data.annotated_image);
      }
      
      if (response.data.plates_detected === 0) {
        setError("No license plates detected in the image. Please try another image.");
      }
    } catch (err) {
      setError(
        err.response?.data?.detail ||
          "Analysis failed. Please check if the backend is running."
      );
      console.error("Analysis error:", err);
    } finally {
      setLoading(false);
    }
  };

  const clearAll = () => {
    setCapturedImage(null);
    setUploadedFile(null);
    setPreviewUrl(null);
    setAnnotatedImageUrl(null);
    setResults(null);
    setError(null);
    if (cameraOn) stopCamera();
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-50 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-blue-600 rounded-lg">
                <Car className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-semibold text-gray-900">
                  License Plate Recognition
                </h1>
                <p className="text-sm text-gray-500">
                  AI-Powered Detection & Analysis
                </p>
              </div>
            </div>

            {apiStatus && (
              <div
                className={`flex items-center space-x-2 px-3 py-1.5 rounded-full ${
                  apiStatus.ready
                    ? "bg-green-50 text-green-700 border border-green-200"
                    : "bg-red-50 text-red-700 border border-red-200"
                }`}
              >
                <div
                  className={`w-2 h-2 rounded-full ${
                    apiStatus.ready ? "bg-green-600" : "bg-red-600"
                  } animate-pulse`}
                />
                <span className="text-sm font-medium">
                  {apiStatus.ready ? "API Ready" : "API Offline"}
                </span>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Panel - Camera/Upload */}
          <div className="space-y-6">
            {/* Camera Feed */}
            <div className="bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden">
              <div className="p-6">
                <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                  <Camera className="w-5 h-5 mr-2 text-blue-600" />
                  {annotatedImageUrl ? "Detection Result" : "Camera Feed"}
                </h2>

                <div className="relative aspect-video bg-gray-100 rounded-lg overflow-hidden border border-gray-200">
                  {cameraOn ? (
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      className="w-full h-full object-cover"
                    />
                  ) : annotatedImageUrl ? (
                    <img
                      src={annotatedImageUrl}
                      alt="Detection Result"
                      className="w-full h-full object-contain"
                    />
                  ) : previewUrl ? (
                    <img
                      src={previewUrl}
                      alt="Preview"
                      className="w-full h-full object-contain"
                    />
                  ) : (
                    <div className="w-full h-full flex flex-col items-center justify-center text-gray-400">
                      <Video className="w-16 h-16 mb-4" />
                      <p className="text-sm">No camera feed or image</p>
                    </div>
                  )}

                  {results?.results?.length > 0 && annotatedImageUrl && (
                    <div className="absolute top-4 left-4 bg-green-600 text-white px-3 py-1.5 rounded-full text-sm font-medium flex items-center shadow-lg">
                      <CheckCircle className="w-4 h-4 mr-1.5" />
                      {results.plates_detected} Plate(s) Detected
                    </div>
                  )}
                </div>

                <canvas ref={canvasRef} className="hidden" />

                <div className="mt-4 flex gap-3">
                  {!cameraOn ? (
                    <button
                      onClick={startCamera}
                      className="flex-1 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2.5 rounded-lg font-medium flex items-center justify-center transition-colors shadow-sm"
                    >
                      <Camera className="w-5 h-5 mr-2" />
                      Start Camera
                    </button>
                  ) : (
                    <>
                      <button
                        onClick={captureFrame}
                        className="flex-1 bg-green-600 hover:bg-green-700 text-white px-4 py-2.5 rounded-lg font-medium flex items-center justify-center transition-colors shadow-sm"
                      >
                        <ZoomIn className="w-5 h-5 mr-2" />
                        Capture Frame
                      </button>
                      <button
                        onClick={stopCamera}
                        className="px-4 py-2.5 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium flex items-center transition-colors shadow-sm"
                      >
                        <X className="w-5 h-5" />
                      </button>
                    </>
                  )}
                </div>
              </div>
            </div>

            {/* Upload Section */}
            <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <Upload className="w-5 h-5 mr-2 text-blue-600" />
                Upload Image
              </h2>

              <div
                onClick={() => fileInputRef.current?.click()}
                className="border-2 border-dashed border-gray-300 hover:border-blue-500 rounded-lg p-8 text-center cursor-pointer transition-colors bg-gray-50"
              >
                <ImageIcon className="w-12 h-12 mx-auto mb-3 text-gray-400" />
                <p className="text-gray-700 font-medium mb-1">
                  Click to upload image
                </p>
                <p className="text-sm text-gray-500">
                  or drag and drop (PNG, JPG, JPEG)
                </p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileUpload}
                  className="hidden"
                />
              </div>

              <div className="mt-6 flex gap-3">
                <button
                  onClick={analyzeImage}
                  disabled={loading || (!capturedImage && !uploadedFile)}
                  className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 disabled:text-gray-500 text-white px-6 py-2.5 rounded-lg font-medium flex items-center justify-center transition-colors shadow-sm disabled:cursor-not-allowed"
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-5 h-5 mr-2" />
                      Analyze Plate
                    </>
                  )}
                </button>

                <button
                  onClick={clearAll}
                  className="px-6 py-2.5 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg font-medium flex items-center transition-colors border border-gray-300"
                >
                  <X className="w-5 h-5 mr-2" />
                  Clear
                </button>
              </div>
            </div>

            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start">
                <AlertCircle className="w-5 h-5 text-red-600 mt-0.5 mr-3 flex-shrink-0" />
                <p className="text-red-800 text-sm">{error}</p>
              </div>
            )}
          </div>

          {/* Right Panel - Results */}
          <div className="space-y-6">
            {results?.results?.length > 0 ? (
              results.results.map((plate, index) => (
                <DetectionCard key={index} plate={plate} index={index} />
              ))
            ) : (
              <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-12">
                <div className="text-center text-gray-400">
                  <FileText className="w-16 h-16 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-gray-600 mb-2">
                    No Results Yet
                  </h3>
                  <p className="text-sm text-gray-500">
                    Capture or upload an image to start detection
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

// Detection Result Card Component - Minimal Google-like Design
function DetectionCard({ plate, index }) {
  return (
    <div className="bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden">
      {/* Header */}
      <div className="bg-blue-600 px-6 py-3">
        <h3 className="text-white font-semibold text-base flex items-center">
          <Shield className="w-5 h-5 mr-2" />
          Detection #{index + 1}
        </h3>
      </div>

      <div className="p-6 space-y-6">
        {/* Plate Number */}
        <div className="bg-gray-50 rounded-lg p-6 text-center border border-gray-200">
          <p className="text-gray-500 text-sm mb-2 font-medium">License Plate Number</p>
          <p className="text-3xl font-bold text-gray-900 tracking-wider font-mono">
            {plate.plate_text || "NO_TEXT"}
          </p>
        </div>

        {/* Confidence Scores */}
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-600 text-sm font-medium">YOLO Confidence</span>
              <span className="text-green-600 font-semibold">
                {((plate.yolo_confidence || 0) * 100).toFixed(1)}%
              </span>
            </div>
            <div className="bg-gray-200 rounded-full h-2 overflow-hidden">
              <div
                className="bg-green-600 h-full transition-all"
                style={{ width: `${(plate.yolo_confidence || 0) * 100}%` }}
              />
            </div>
          </div>

          <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-600 text-sm font-medium">OCR Confidence</span>
              <span className="text-blue-600 font-semibold">
                {((plate.ocr_confidence || 0) * 100).toFixed(1)}%
              </span>
            </div>
            <div className="bg-gray-200 rounded-full h-2 overflow-hidden">
              <div
                className="bg-blue-600 h-full transition-all"
                style={{ width: `${(plate.ocr_confidence || 0) * 100}%` }}
              />
            </div>
          </div>
        </div>

        {/* Bounding Box Info */}
        {plate.box && (
          <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
            <p className="text-gray-600 text-sm font-medium mb-2">Detection Region</p>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="text-gray-700">
                <span className="text-gray-500">Width:</span> {plate.box.width || 0}px
              </div>
              <div className="text-gray-700">
                <span className="text-gray-500">Height:</span> {plate.box.height || 0}px
              </div>
            </div>
          </div>
        )}

        {/* Vehicle Insights */}
        {plate.vehicle_insights && (
          <div className="bg-blue-50 rounded-lg p-6 border border-blue-200">
            <h4 className="text-gray-900 font-semibold mb-4 flex items-center">
              <Sparkles className="w-5 h-5 mr-2 text-blue-600" />
              AI-Powered Insights
            </h4>

            {plate.vehicle_insights.raw_analysis ? (
              <div className="space-y-3">
                <div className="bg-white rounded-lg p-4 border border-gray-200">
                  <div className="markdown-content">
                    <ReactMarkdown
                      components={{
                        p: ({node, ...props}) => <p className="text-gray-700 text-sm mb-2 leading-relaxed" {...props} />,
                        strong: ({node, ...props}) => <strong className="text-gray-900 font-semibold" {...props} />,
                        em: ({node, ...props}) => <em className="text-gray-700 italic" {...props} />,
                        h1: ({node, ...props}) => <h1 className="text-gray-900 text-lg font-bold mb-2" {...props} />,
                        h2: ({node, ...props}) => <h2 className="text-gray-900 text-base font-semibold mb-2" {...props} />,
                        h3: ({node, ...props}) => <h3 className="text-gray-900 text-sm font-semibold mb-1" {...props} />,
                        ol: ({node, ...props}) => <ol className="list-decimal ml-4 space-y-1 text-gray-700 text-sm" {...props} />,
                        ul: ({node, ...props}) => <ul className="list-disc ml-4 space-y-1 text-gray-700 text-sm" {...props} />,
                        li: ({node, ...props}) => <li className="text-gray-700 text-sm" {...props} />,
                        code: ({node, inline, ...props}) => 
                          inline ? (
                            <code className="text-blue-600 bg-blue-50 px-1 rounded text-xs border border-blue-200" {...props} />
                          ) : (
                            <code className="block text-blue-600 bg-blue-50 p-2 rounded text-xs border border-blue-200" {...props} />
                          ),
                        pre: ({node, ...props}) => <pre className="bg-gray-100 p-2 rounded overflow-x-auto border border-gray-300" {...props} />,
                        blockquote: ({node, ...props}) => <blockquote className="border-l-4 border-blue-500 pl-4 italic text-gray-600" {...props} />,
                      }}
                    >
                      {plate.vehicle_insights.raw_analysis}
                    </ReactMarkdown>
                  </div>
                </div>
                <div className="flex items-center text-xs text-gray-500">
                  <Car className="w-3 h-3 mr-1" />
                  Powered by {plate.vehicle_insights.ai_provider || "AI"}
                </div>
              </div>
            ) : (
              <p className="text-gray-600 text-sm">
                {plate.vehicle_insights.error || "No insights available"}
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;