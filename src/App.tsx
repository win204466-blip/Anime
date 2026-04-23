import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils, VRM } from '@pixiv/three-vrm';
import { FaceLandmarker, HandLandmarker, PoseLandmarker, FilesetResolver, DrawingUtils } from '@mediapipe/tasks-vision';
import * as Kalidokit from 'kalidokit';

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const [loading, setLoading] = useState<string>('Initializing...');

  useEffect(() => {
    let isMounted = true;
    let faceLandmarker: FaceLandmarker;
    let handLandmarker: HandLandmarker;
    let poseLandmarker: PoseLandmarker;
    let vrm: VRM | undefined;
    let camera: THREE.PerspectiveCamera;
    let scene: THREE.Scene;
    let renderer: THREE.WebGLRenderer;
    let animationFrameId: number;
    
    let lastVideoTime = -1;

    const init = async () => {
      setLoading('Loading Auto-trackers (Face & Hands)...');
      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.12/wasm'
      );
      
      faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
          delegate: 'GPU'
        },
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: true,
        runningMode: 'VIDEO',
        numFaces: 1
      });

      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
          delegate: 'GPU'
        },
        runningMode: 'VIDEO',
        numHands: 2
      });

      poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
          delegate: 'GPU'
        },
        runningMode: 'VIDEO',
        outputSegmentationMasks: false
      });

      setLoading('Loading 3D Model...');
      
      // Setup ThreeJS
      scene = new THREE.Scene();
      
      camera = new THREE.PerspectiveCamera(35, window.innerWidth / window.innerHeight, 0.1, 20);
      camera.position.set(0, 1.0, 2.3); // Focus on upper body and hands
      
      renderer = new THREE.WebGLRenderer({ canvas: canvasRef.current!, alpha: true, antialias: true });
      renderer.setSize(window.innerWidth, window.innerHeight);
      renderer.setPixelRatio(window.devicePixelRatio);
      renderer.outputColorSpace = THREE.SRGBColorSpace;

      // Light
      const light = new THREE.DirectionalLight(0xffffff, Math.PI);
      light.position.set(1, 1, 1).normalize();
      scene.add(light);
      const ambientLight = new THREE.AmbientLight(0xffffff, Math.PI * 0.5);
      scene.add(ambientLight);

      // Load VRM
      const loader = new GLTFLoader();
      loader.register((parser) => new VRMLoaderPlugin(parser));

      try {
        // VRM sample model from uniVRM
        const gltf = await loader.loadAsync('https://raw.githubusercontent.com/vrm-c/UniVRM/master/Tests/Models/Alicia_vrm-0.51/AliciaSolid_vrm-0.51.vrm');
        const vrmData = gltf.userData.vrm;
        
        VRMUtils.removeUnnecessaryJoints(gltf.scene);
        vrm = vrmData;
        scene.add(vrm!.scene);
        
        // Face the camera
        vrm!.scene.rotation.y = Math.PI;

        setLoading('Starting Camera...');
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: 'user' }
        });
        
        if (isMounted && videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play().catch(e => console.warn('Play interrupted:', e));
          setLoading('');
        }
      } catch (err) {
        console.error(err);
        setLoading('Failed to load VRM or Camera.');
      }
    };

    const clock = new THREE.Clock();

    // Registries for smooth interpolation running at 60fps
    const targetRotations: Record<string, THREE.Quaternion> = {};
    const targetExpressions: Record<string, number> = {};

    const lerp = (start: number, end: number, t: number) => {
       return start * (1 - t) + end * t;
    };

    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);

      if (videoRef.current && videoRef.current.readyState >= 2 && faceLandmarker && handLandmarker && poseLandmarker) {
        const timeNow = performance.now();
        if (lastVideoTime !== videoRef.current.currentTime) {
          const result = faceLandmarker.detectForVideo(videoRef.current, timeNow);
          const handResult = handLandmarker.detectForVideo(videoRef.current, timeNow);
          const poseResult = poseLandmarker.detectForVideo(videoRef.current, timeNow);
          lastVideoTime = videoRef.current.currentTime;
          
          if (vrm && result.faceLandmarks && result.faceLandmarks.length > 0) {
            
            // Let Kalidokit solve the face
            const kalidoFace = Kalidokit.Face.solve(result.faceLandmarks[0], {
              runtime: "mediapipe",
              video: videoRef.current
            });

            if (kalidoFace) {
               const targetRotation = new THREE.Euler(
                 kalidoFace.head.x, 
                 kalidoFace.head.y, 
                 kalidoFace.head.z
               );
               targetRotations['head'] = new THREE.Quaternion().setFromEuler(targetRotation);
              
               const targetNeck = new THREE.Euler(
                 kalidoFace.head.x * 0.5, 
                 kalidoFace.head.y * 0.5, 
                 kalidoFace.head.z * 0.5
               );
               targetRotations['neck'] = new THREE.Quaternion().setFromEuler(targetNeck);

              // Blendshapes mapped into targets
              targetExpressions['blinkLeft'] = kalidoFace.eye.l;
              targetExpressions['blinkRight'] = kalidoFace.eye.r;
              targetExpressions['aa'] = kalidoFace.mouth.shape.A;
              targetExpressions['ee'] = kalidoFace.mouth.shape.E;
              targetExpressions['ih'] = kalidoFace.mouth.shape.I;
              targetExpressions['oh'] = kalidoFace.mouth.shape.O;
              targetExpressions['ou'] = kalidoFace.mouth.shape.U;
              targetExpressions['joy'] = kalidoFace.mouth.y > 0.05 ? kalidoFace.mouth.y * 3.0 : 0;
              
              // Map pupils
              const eyeRotation = new THREE.Euler(
                kalidoFace.pupil.y,
                kalidoFace.pupil.x,
                0
              );
              targetRotations['rightEye'] = new THREE.Quaternion().setFromEuler(eyeRotation);
              targetRotations['leftEye'] = new THREE.Quaternion().setFromEuler(eyeRotation);
            }
            
            // Draw mesh on overlay
            if (overlayCanvasRef.current && result.faceLandmarks && result.faceLandmarks.length > 0) {
              const canvas = overlayCanvasRef.current;
              const ctx = canvas.getContext('2d');
              if (ctx && videoRef.current) {
                if (canvas.width !== videoRef.current.videoWidth) {
                  canvas.width = videoRef.current.videoWidth;
                  canvas.height = videoRef.current.videoHeight;
                }
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                const drawingUtils = new DrawingUtils(ctx);
                const landmarks = result.faceLandmarks[0];
                
                drawingUtils.drawConnectors(
                  landmarks, 
                  FaceLandmarker.FACE_LANDMARKS_TESSELATION, 
                  { color: '#ffffff40', lineWidth: 1 }
                );
                drawingUtils.drawConnectors(
                  landmarks, 
                  FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, 
                  { color: '#ff3030', lineWidth: 2 }
                );
                drawingUtils.drawConnectors(
                  landmarks, 
                  FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, 
                  { color: '#30ff30', lineWidth: 2 }
                );
                drawingUtils.drawConnectors(
                  landmarks, 
                  FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, 
                  { color: '#e0e0e0', lineWidth: 2 }
                );
                drawingUtils.drawConnectors(
                  landmarks, 
                  FaceLandmarker.FACE_LANDMARKS_LIPS, 
                  { color: '#e0e0e0', lineWidth: 2 }
                );
              }
            } else if (overlayCanvasRef.current) {
              const canvas = overlayCanvasRef.current;
              const ctx = canvas.getContext('2d');
              if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
            }
          }
          
          // Handle Pose tracking (Arms, Shoulders, Spine)
          if (vrm && poseResult.landmarks && poseResult.worldLandmarks && poseResult.landmarks.length > 0) {
            if (overlayCanvasRef.current) {
              const canvas = overlayCanvasRef.current;
              const ctx = canvas.getContext('2d');
              if (ctx) {
                const drawingUtils = new DrawingUtils(ctx);
                drawingUtils.drawConnectors(poseResult.landmarks[0], PoseLandmarker.POSE_CONNECTIONS, { color: '#00FFFF', lineWidth: 2 });
                // We don't need to draw every landmark to avoid clutter 
              }
            }

            // Let Kalidokit solve the pose
            const kalidoPose = Kalidokit.Pose.solve(poseResult.worldLandmarks[0], poseResult.landmarks[0], {
              runtime: "mediapipe",
              video: videoRef.current
            });

            if (kalidoPose) {
              // Apply Kalidokit Pose Rig
              
              const vrmBoneMap: Record<string, string> = {
                spine: 'spine',
                chest: 'chest',
                upperChest: 'upperChest',
                neck: 'neck',
                leftShoulder: 'leftShoulder',
                leftUpperArm: 'leftUpperArm',
                leftLowerArm: 'leftLowerArm',
                leftHand: 'leftHand',
                rightShoulder: 'rightShoulder',
                rightUpperArm: 'rightUpperArm',
                rightLowerArm: 'rightLowerArm',
                rightHand: 'rightHand'
              };

              // Iterate over all calculated parts (RightUpperArm, Spine, etc.)
              for (const [key, partValue] of Object.entries(kalidoPose)) {
                // Kalidokit keys are like 'RightUpperArm', 'Spine', etc.
                const vrmBoneName = key.charAt(0).toLowerCase() + key.slice(1); // 'rightUpperArm'
                if (vrmBoneMap[vrmBoneName]) {
                   // Support for standard rotations {x, y, z} vs Hips object {rotation: {x, y, z}, position: ...}
                   const r = (partValue as any).rotation || partValue;
                   
                   if (r && typeof r.x === 'number' && !isNaN(r.x)) {
                     targetRotations[vrmBoneName] = new THREE.Quaternion().setFromEuler(new THREE.Euler(r.x, r.y, r.z));
                   }
                }
              }
            }
          }

          // Handle Hand Tracking for Fingers
          if (vrm) {
            if (handResult.landmarks && handResult.landmarks.length > 0) {
              handResult.landmarks.forEach((landmarks, index) => {
                const handedness = handResult.handednesses[index][0].categoryName;
                
                // Draw hand landmarks
                if (overlayCanvasRef.current) {
                  const canvas = overlayCanvasRef.current;
                  const ctx = canvas.getContext('2d');
                  if (ctx) {
                    const drawingUtils = new DrawingUtils(ctx);
                    drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, { color: '#00FF00', lineWidth: 2 });
                  }
                }

                // Finger tracking using Kalidokit
                const kalidoHand = Kalidokit.Hand.solve(landmarks, handedness);
                if (kalidoHand) {
                  // Kalidokit outputs keys based on the exact string passed in (handedness)
                  const prefix = handedness; 
                  // But for VRM, mediaPipe's 'Left' actually corresponds to the user's Right hand 
                  // (since camera is mirrored). VRM expects 'right', 'left' lowercased.
                  const vrmPrefix = handedness === 'Left' ? 'right' : 'left';
                  const fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Little'];
                  const joints = ['Proximal', 'Intermediate', 'Distal'];

                  fingers.forEach(finger => {
                    joints.forEach(joint => {
                      const key = `${prefix}${finger}${joint}`;
                      const rotation = (kalidoHand as any)[key];
                      
                      const boneName = `${vrmPrefix}${finger}${joint}`;
                      
                      if (rotation) {
                         targetRotations[boneName] = new THREE.Quaternion().setFromEuler(new THREE.Euler(rotation.x, rotation.y, rotation.z));
                      }
                    });
                  });
                }
              });
            }
          }
        }
      }

      if (vrm) {
        // Apply smooth interpolations to all targets
        const smoothFactor = 0.15; // Smooths out the 30fps web cam into 60fps+ rendering 

        // 1. Bones interpolation
        for (const [boneName, targetQuat] of Object.entries(targetRotations)) {
          const bone = vrm.humanoid?.getNormalizedBoneNode(boneName as any);
          if (bone) {
             bone.quaternion.slerp(targetQuat, smoothFactor);
          }
        }
        
        // 2. Expressions interpolation
        for (const [expName, targetValue] of Object.entries(targetExpressions)) {
          const current = vrm.expressionManager?.getValue(expName) || 0;
          vrm.expressionManager?.setValue(expName, lerp(current, targetValue, smoothFactor));
        }

        vrm.update(clock.getDelta());
      }
      
      renderer.render(scene, camera);
    };

    init().then(() => {
      animate();
    });

    const handleResize = () => {
      if (camera && renderer) {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      isMounted = false;
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animationFrameId);
      if (faceLandmarker) faceLandmarker.close();
      if (videoRef.current && videoRef.current.srcObject) {
         (videoRef.current.srcObject as MediaStream).getTracks().forEach(t => t.stop());
      }
    };
  }, []);

  return (
    <div className="relative w-full h-screen overflow-hidden bg-gradient-to-b from-slate-900 via-indigo-950 to-purple-950">
      {/* Streamer VTuber Decorative Room Elements */}
      <div className="absolute inset-0 opacity-20 bg-[linear-gradient(rgba(255,255,255,0.1)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.1)_1px,transparent_1px)] bg-[size:40px_40px]"></div>
      <div className="absolute top-[20%] left-[20%] w-96 h-96 bg-pink-500 rounded-full mix-blend-screen filter blur-[128px] animate-pulse"></div>
      <div className="absolute bottom-[20%] right-[20%] w-96 h-96 bg-blue-500 rounded-full mix-blend-screen filter blur-[128px] animate-pulse" style={{ animationDelay: '2s' }}></div>

      {loading && (
        <div className="absolute inset-0 z-50 flex flex-col items-center justify-center bg-black/80 text-white">
          <div className="w-16 h-16 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin mb-4"></div>
          <p className="text-lg font-medium animate-pulse">{loading}</p>
        </div>
      )}
      
      <canvas
        ref={canvasRef}
        className="w-full h-full object-cover relative z-10"
      />
      
      <div className="absolute bottom-4 left-4 w-40 md:w-64 rounded-xl overflow-hidden shadow-2xl border-4 border-indigo-500/30 backdrop-blur-sm bg-black/60 z-40">
        <video
          ref={videoRef}
          className="w-full h-auto transform -scale-x-100 object-cover"
          playsInline
          muted
          autoPlay
        />
        <canvas
          ref={overlayCanvasRef}
          className="w-full h-auto transform -scale-x-100 object-cover absolute top-0 left-0 pointer-events-none"
        />
        <div className="absolute bottom-2 left-0 right-0 text-center text-xs text-indigo-200 font-bold tracking-widest z-10 drop-shadow-md">
          CAMERA FEED 
        </div>
      </div>
    </div>
  );
}
