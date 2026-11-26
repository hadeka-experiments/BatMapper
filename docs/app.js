/* BatMapper Web — browser port of the iOS app (MBatmaN).
   Same pipeline: chirp emission -> bandpass -> cross-correlation ->
   smoothing -> peak detection -> distance candidates -> map points. */
"use strict";

// ---------- Signal processing (port of SignalProcessing.swift) ----------
const CHIRP_48K = [
    0, -0.0026132, -0.0042297, 0.034795, -0.059418,
    0.02043366, 0.0929047, -0.195618, 0.1606969, 0.0527688,
    -0.3189, 0.412045, -0.1913417, -0.248626, 0.60615,
    -0.584082, 0.130236, 0.487933, -0.84944, 0.675767,
    -0.040697, -0.663296, 0.9829, -0.705999, -3.919488e-15,
    0.70215938, -0.98290, 0.696682, -0.040697, -0.589379,
    0.849443, -0.63949, 0.130236, 0.369866, -0.60615,
    0.50764897, -0.19134, -0.13862, 0.3189, -0.304114,
    0.1606969, -0.000533, -0.0929047, 0.10128, -0.059418,
    0.01542, 0.00422975, -0.00339, 0
];
const COEFF_A = [1.0, -5.3191965, 12.7014, -17.702244, 15.70216,
    -9.05636137, 3.31085209, -0.70045, 0.0655805];
const COEFF_B = [0.2560869, -2.048695, 7.1704337, -14.3408674, 17.926084266,
    -14.3408674, 7.1704337, -2.048695, 0.2560869];
const N_CANDIDATES = 8, MIN_RANGE = 0.3, MAX_RANGE = 5.0;

function resampleLinear(src, srcRate, dstRate) {
    const n = Math.max(2, Math.round(src.length * dstRate / srcRate));
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        const pos = i * srcRate / dstRate;
        const i0 = Math.floor(pos), i1 = Math.min(src.length - 1, i0 + 1);
        out[i] = src[i0] + (src[i1] - src[i0]) * (pos - i0);
    }
    return out;
}

class SignalProcessing {
    constructor(rate) {
        this.rate = rate;
        this.overLap = Math.round(40 * rate / 1000);
        this.chirp = rate === 48000 ? Float32Array.from(CHIRP_48K)
            : resampleLinear(CHIRP_48K, 48000, rate);
    }
    process(rec) { // rec: Int16Array
        if (rec.length <= this.chirp.length + 10)
            return Array.from({ length: N_CANDIDATES }, () => [0, 0]);
        let f = this.bandPass(rec);
        f = this.crossCorrelate(f, this.chirp);
        this.smooth(f, 5);
        const begin = this.findBeginning(f, Math.min(this.overLap, f.length));
        return this.distanceGeneration(begin, f);
    }
    bandPass(x) {
        const n = x.length, a = COEFF_A, b = COEFF_B, y = new Float32Array(n);
        for (let i = 0; i < n - 1; i++) {
            let v = b[0] * x[i];
            const jMax = Math.min(i, a.length - 1);
            for (let j = 1; j <= jMax; j++) v += b[j] * x[i - j] - a[j] * y[i - j];
            y[i] = v;
        }
        return y;
    }
    crossCorrelate(f, g) {
        const n = f.length, m = g.length, res = new Float32Array(n);
        for (let T = 0; T < n - m; T++) {
            let s = 0;
            for (let t = 0; t < m; t++) s += f[t + T] * g[t];
            res[T] = s;
        }
        return res;
    }
    smooth(buf, amount) {
        const n = buf.length, temp = Float32Array.from(buf);
        for (let i = 0; i < n; i++) {
            let s = 0;
            const j0 = Math.max(0, i - 2 * amount), j1 = Math.min(n - 1, i + 2 * amount);
            for (let j = j0; j <= j1; j++) {
                const d = (j - i) / amount;
                s += Math.abs(temp[j]) * Math.exp(-d * d / 2);
            }
            buf[i] = s;
        }
    }
    findBeginning(rec, limit) {
        let maxVal = 0, idx = 0;
        for (let T = 0; T < limit; T++)
            if (Math.abs(rec[T]) > maxVal) { maxVal = Math.abs(rec[T]); idx = T; }
        return idx;
    }
    distanceGeneration(begin, rec) {
        const rate = this.rate;
        const minPts = Math.floor(2 * MIN_RANGE / 346 * rate);
        const maxPts = Math.floor(2 * MAX_RANGE / 346 * rate);
        const dis = [], amp = [];
        const res = Array.from({ length: N_CANDIDATES }, () => [0, 0]);
        const s0 = begin + minPts, s1 = Math.min(begin + maxPts, rec.length - 3);
        for (let i = Math.max(s0, 2); i < s1; i++) {
            let localMax = true;
            for (let j = -2; j < 2; j++) if (rec[i + j] > rec[i]) { localMax = false; break; }
            if (localMax) { dis.push(173.0 * (i - begin) / rate); amp.push(rec[i]); }
        }
        if (!dis.length) return res;
        const ampCopy = amp.slice();
        for (let k = 0; k < Math.min(N_CANDIDATES, dis.length); k++) {
            let mi = 0;
            for (let i = 0; i < ampCopy.length; i++) if (ampCopy[i] > ampCopy[mi]) mi = i;
            res[k] = [dis[mi], ampCopy[mi]];
            ampCopy[mi] = -Infinity;
        }
        return res;
    }
}

// ---------- Audio engine (port of AudioEngine.swift) ----------
class SonarEngine {
    constructor(onDistances) {
        this.onDistances = onDistances;
        this.distL = 0; this.distR = 0;
        this.running = false;
    }
    async start() {
        const AC = window.AudioContext || window.webkitAudioContext;
        this.ctx = new AC({ sampleRate: 48000 });
        if (this.ctx.state === "suspended") await this.ctx.resume();
        const rate = this.ctx.sampleRate;
        this.rate = rate;
        this.sig = new SignalProcessing(rate);
        this.windowSize = Math.round(2 * 70 * rate / 1000); // 2 * 70ms
        this.micBuf = new Int16Array(this.windowSize);
        this.micFill = 0;

        // Chirp loop: template + 40ms silence, looped like the iOS AVAudioPlayer loop
        const chirp = this.sig.chirp;
        const silence = Math.round(0.04 * rate);
        let peak = 0; for (const v of chirp) peak = Math.max(peak, Math.abs(v));
        const loop = this.ctx.createBuffer(1, chirp.length + silence, rate);
        const data = loop.getChannelData(0);
        for (let i = 0; i < chirp.length; i++) data[i] = chirp[i] / peak * 0.9;
        this.chirpSrc = this.ctx.createBufferSource();
        this.chirpSrc.buffer = loop;
        this.chirpSrc.loop = true;
        this.chirpSrc.connect(this.ctx.destination);
        this.chirpSrc.start();

        // Mic capture
        this.stream = await navigator.mediaDevices.getUserMedia({
            audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false }
        });
        this.micNode = this.ctx.createMediaStreamSource(this.stream);
        this.proc = this.ctx.createScriptProcessor(2048, 1, 1);
        this.proc.onaudioprocess = (e) => this.onAudio(e.inputBuffer);
        this.micNode.connect(this.proc);
        this.proc.connect(this.ctx.destination); // required for onaudioprocess
        this.running = true;
    }
    onAudio(buf) {
        if (!this.running) return;
        const ch = buf.getChannelData(0);
        for (let i = 0; i < ch.length; i++) {
            let v = Math.round(Math.max(-1, Math.min(1, ch[i])) * 32767);
            this.micBuf[this.micFill++] = v;
            if (this.micFill >= this.windowSize) {
                this.micFill = 0;
                this.processWindow(this.micBuf);
            }
        }
    }
    processWindow(rec) {
        const cands = this.sig.process(rec);
        let dL = cands[0][0], dR = cands[0][0];
        // Same distance-filtering heuristics as the app
        for (const c of cands)
            if (c[0] < 0.95 && c[0] > 0.7 && Math.abs(c[0] - this.distR) > 0.2) { dL = c[0]; break; }
        for (const c of cands)
            if (c[0] > 1.3 && c[0] < 1.5 && Math.abs(c[0] - this.distL) > 0.2) { dR = c[0]; break; }
        this.distL = dL; this.distR = dR;
        if (this.onDistances) this.onDistances(dL, dR);
    }
    stop() {
        this.running = false;
        try { this.chirpSrc && this.chirpSrc.stop(); } catch (e) {}
        try { this.proc && this.proc.disconnect(); } catch (e) {}
        try { this.micNode && this.micNode.disconnect(); } catch (e) {}
        if (this.stream) this.stream.getTracks().forEach(t => t.stop());
        if (this.ctx) this.ctx.close();
    }
}

// ---------- Kalman filter (port of KalmanFilter.swift) ----------
class KalmanFilter {
    constructor(q = 0.01, r = 0.1, p = 0.1) {
        this.q = q; this.r = r; this.p0 = p; this.reset();
    }
    reset() { this.x = 0; this.p = this.p0; this.k = 0; this.init = false; }
    filter(m) {
        if (!this.init) { this.x = m; this.init = true; return this.x; }
        this.p += this.q;
        this.k = this.p / (this.p + this.r);
        this.x = this.x + this.k * (m - this.x);
        this.p = (1 - this.k) * this.p;
        return this.x;
    }
}

// ---------- Loop closure (port of InertialTracking.swift) ----------
function loopClosure(trace, wallCam, wallMic) {
    if (trace.length <= 4) return;
    const numPoints = trace.length / 2;
    const errX = trace[trace.length - 2], errY = trace[trace.length - 1];
    for (let k = 1; k < numPoints; k++) {
        const ratio = k / (numPoints - 1);
        const iX = 2 * k, iY = 2 * k + 1;
        const cX = errX * ratio, cY = errY * ratio;
        trace[iX] -= cX; trace[iY] -= cY;
        if (iX < wallCam.length) wallCam[iX] -= cX;
        if (iY < wallCam.length) wallCam[iY] -= cY;
        if (iX < wallMic.length) wallMic[iX] -= cX;
        if (iY < wallMic.length) wallMic[iY] -= cY;
    }
}

// ---------- Motion tracker (port of MotionTracker.swift) ----------
class MotionTracker {
    constructor() {
        this.stepLength = 1.0;
        this.sensorScanRate = 20;
        this.walkingTimeout = 40;
        this.doorDepth = 0.05; this.doorCountMax = 20; this.doorCountMin = 4;
        this.doorWindow = 50;
        this.yawFilter = new KalmanFilter(0.1, 0.1, 1.0);
        this.dCamFilter = new KalmanFilter(0.01, 0.1, 1.0);
        this.dMicFilter = new KalmanFilter(0.01, 0.1, 1.0);
        this.reset();
    }
    reset() {
        this.yaw = 0; this.isTracking = false;
        this.trace = []; this.wallCam = []; this.wallMic = []; this.doors = [];
        this.pointCount = 0;
        this.isWalking = false; this.walkingTicks = 0; this.idleTicks = 0;
        this.queueOrientation = []; this.rawReading = [];
        this.mean = 0; this.sum = 0; this.drift = 0; this.delay = 0;
        this.diffInputMean = 0; this.turnDrift = 0;
        this.dCamBuffer = []; this.dMicBuffer = [];
        this.dCamDoorBuffer = []; this.dMicDoorBuffer = [];
        this.inDoorCam = false; this.inDoorMic = false;
        this.camCount = 0; this.micCount = 0;
        this.doorLQ = []; this.doorRQ = [];
        this.yawFilter.reset(); this.dCamFilter.reset(); this.dMicFilter.reset();
    }
    // One sensor tick. rawYawDeg: compass yaw in degrees; dCam/dMic: wall
    // distances (0 = none); accelMag: user-acceleration magnitude (m/s^2).
    update(rawYawDeg, dCamRaw, dMicRaw, accelMag) {
        if (!this.isTracking) return;
        const smoothYaw = this.smoothOrientation(rawYawDeg);
        this.yaw = this.yawFilter.filter(smoothYaw);

        let dCam = dCamRaw, dMic = dMicRaw;
        if (dCam !== 0) dCam = this.dCamFilter.filter(dCam); else this.dCamFilter.reset();
        if (dMic !== 0) dMic = this.dMicFilter.filter(dMic); else this.dMicFilter.reset();

        if (this.doorLQ.length < this.doorWindow) this.doorLQ.push(dCam);
        else { this.doorLQ.shift(); this.doorLQ.push(dCam); }
        if (this.doorRQ.length < this.doorWindow) this.doorRQ.push(dMic);
        else { this.doorRQ.shift(); this.doorRQ.push(dMic); }

        // Walking detection: 0.05 g ~= 0.49 m/s^2
        if (accelMag > 0.49) {
            this.isWalking = true;
            this.walkingTicks = Math.max(this.walkingTicks, Math.floor(this.sensorScanRate / 2));
            this.idleTicks = 0;
        }
        if (this.walkingTicks > 0) this.walkingTicks--;
        else if (this.isWalking) {
            this.idleTicks++;
            if (this.idleTicks > this.walkingTimeout) this.isWalking = false;
        }

        if (this.trace.length === 0) {
            this.trace.push(0, 0); this.wallCam.push(0, 0); this.wallMic.push(0, 0);
            this.dCamBuffer.push(dCam, dCam, dCam); this.dMicBuffer.push(dMic, dMic, dMic);
            this.pointCount = 1;
        } else if (this.isWalking || this.walkingTicks > 0) {
            const yawRad = this.yaw * Math.PI / 180;
            const disp = this.stepLength / this.sensorScanRate;
            const x = this.trace[this.trace.length - 2] + Math.cos(yawRad) * disp;
            const y = this.trace[this.trace.length - 1] + Math.sin(yawRad) * disp;
            let xCam = 0, yCam = 0, xMic = 0, yMic = 0;
            const tracingDelay = 5;
            if (this.trace.length > tracingDelay) {
                const negYawRad = -this.yaw * Math.PI / 180 + Math.PI;
                if (dCam !== 0) {
                    xCam = this.trace[this.trace.length - tracingDelay - 1] - dCam * Math.sin(negYawRad);
                    yCam = this.trace[this.trace.length - tracingDelay] - dCam * Math.cos(negYawRad);
                }
                if (dMic !== 0) {
                    xMic = this.trace[this.trace.length - tracingDelay - 1] + dMic * Math.sin(negYawRad);
                    yMic = this.trace[this.trace.length - tracingDelay] + dMic * Math.cos(negYawRad);
                }
            }
            this.trace.push(x, y);
            this.wallCam.push(xCam, yCam); this.wallMic.push(xMic, yMic);
            this.pointCount++;
            this.processDoorDetection(dCam, dMic, xCam, yCam, xMic, yMic);
            this.dCamBuffer.push(dCam); this.dMicBuffer.push(dMic);
            if (this.dCamBuffer.length > 4) { this.dCamBuffer.shift(); this.dMicBuffer.shift(); }
        }
    }
    processDoorDetection(dCam, dMic, xCam, yCam, xMic, yMic) {
        if (this.dCamBuffer.length < 3) return;
        if (dCam - this.dCamBuffer[0] > this.doorDepth &&
            dCam - this.dCamBuffer[1] > this.doorDepth &&
            dCam - this.dCamBuffer[2] > this.doorDepth) this.inDoorCam = true;
        if (dMic - this.dMicBuffer[0] > this.doorDepth &&
            dMic - this.dMicBuffer[1] > this.doorDepth &&
            dMic - this.dMicBuffer[2] > this.doorDepth) this.inDoorMic = true;
        if (this.inDoorCam) { this.dCamDoorBuffer.push(xCam, yCam); this.camCount++; }
        if (this.inDoorMic) { this.dMicDoorBuffer.push(xMic, yMic); this.micCount++; }
        if (this.dCamBuffer[0] - dCam > this.doorDepth &&
            this.camCount < this.doorCountMax && this.camCount > this.doorCountMin &&
            this.dCamBuffer[1] - dCam > this.doorDepth) {
            this.inDoorCam = false;
            this.doors.push(...this.dCamDoorBuffer);
            this.dCamDoorBuffer = []; this.camCount = 0;
        }
        if (this.dMicBuffer[0] - dMic > this.doorDepth &&
            this.micCount < this.doorCountMax && this.micCount > this.doorCountMin &&
            this.dMicBuffer[1] - dMic > this.doorDepth) {
            this.inDoorMic = false;
            this.doors.push(...this.dMicDoorBuffer);
            this.dMicDoorBuffer = []; this.micCount = 0;
        }
        if (this.camCount > this.doorCountMax) { this.dCamDoorBuffer = []; this.camCount = 0; this.inDoorCam = false; }
        if (this.micCount > this.doorCountMax) { this.dMicDoorBuffer = []; this.micCount = 0; this.inDoorMic = false; }
    }
    smoothOrientation(inputRaw) {
        let input = inputRaw;
        const numberOfPoints = 10, orientationTolerance = 5, bufferLength = 10;
        if (this.queueOrientation.length === 0) { this.drift = input; input = 0; }
        else input = input - this.drift;
        if (this.queueOrientation.length < numberOfPoints) {
            this.queueOrientation.push(input);
            this.rawReading.push(input);
            this.sum += input; this.mean = 0;
            return input;
        }
        let sumBuffer = 0;
        const n = this.rawReading.length;
        for (let i = 1; i <= bufferLength; i++) if (n - i >= 0) sumBuffer += this.rawReading[n - i];
        const avg = sumBuffer / bufferLength;
        let sumVar = 0;
        for (let i = 1; i <= bufferLength; i++) if (n - i >= 0) {
            const d = this.rawReading[n - i] - avg; sumVar += d * d;
        }
        const avgVar = sumVar / bufferLength;
        if (avgVar < orientationTolerance) {
            this.queueOrientation.shift();
            this.queueOrientation.push(this.mean);
            this.rawReading.push(input);
        } else this.delay = 120;
        if (this.delay >= 0) {
            this.delay--;
            this.rawReading.push(input);
            this.diffInputMean = input - this.mean;
            const dir = this.diffInputMean > 0 ? 1 : -1;
            const absDiff = Math.abs(this.diffInputMean);
            if (absDiff > 90 - orientationTolerance && absDiff < 90 + orientationTolerance) {
                this.mean += dir * 90;
                this.turnDrift += this.mean - input;
            }
            return input;
        }
        this.mean = input + this.turnDrift;
        return this.mean;
    }
    applyLoopClosure() { loopClosure(this.trace, this.wallCam, this.wallMic); }
}

// ---------- Map canvas (port of MapCanvasView.swift) ----------
class MapCanvas {
    constructor(canvas) {
        this.cv = canvas; this.g = canvas.getContext("2d");
        this.scale = 1; this.offX = 0; this.offY = 0;
        this.autoFit = true;
        this.colors = { trace: "#00f2ff", cam: "#ff8c00", mic: "#6699ff", door: "#ffd900" };
        this.bindGestures();
        this.resize();
        window.addEventListener("resize", () => this.resize());
    }
    resize() {
        const r = this.cv.parentElement.getBoundingClientRect();
        this.cv.width = r.width * devicePixelRatio;
        this.cv.height = r.height * devicePixelRatio;
    }
    bindGestures() {
        let drag = null, pinch = null;
        const pts = new Map();
        this.cv.addEventListener("pointerdown", e => {
            this.cv.setPointerCapture(e.pointerId);
            pts.set(e.pointerId, [e.clientX, e.clientY]);
            if (pts.size === 1) { drag = { x: e.clientX, y: e.clientY, ox: this.offX, oy: this.offY }; this.autoFit = false; }
            if (pts.size === 2) {
                const p = [...pts.values()];
                pinch = { d: Math.hypot(p[0][0]-p[1][0], p[0][1]-p[1][1]), s: this.scale };
                drag = null;
            }
        });
        this.cv.addEventListener("pointermove", e => {
            if (!pts.has(e.pointerId)) return;
            pts.set(e.pointerId, [e.clientX, e.clientY]);
            if (drag) {
                this.offX = drag.ox + (e.clientX - drag.x);
                this.offY = drag.oy + (e.clientY - drag.y);
            } else if (pinch && pts.size === 2) {
                this.autoFit = false;
                const p = [...pts.values()];
                const d = Math.hypot(p[0][0]-p[1][0], p[0][1]-p[1][1]);
                this.scale = Math.max(0.1, Math.min(20, pinch.s * d / pinch.d));
            }
        });
        const up = e => { pts.delete(e.pointerId); drag = null; pinch = null; };
        this.cv.addEventListener("pointerup", up);
        this.cv.addEventListener("pointercancel", up);
        this.cv.addEventListener("wheel", e => {
            e.preventDefault();
            this.autoFit = false;
            this.scale = Math.max(0.1, Math.min(20, this.scale * (e.deltaY < 0 ? 1.1 : 0.9)));
        }, { passive: false });
    }
    draw(t) {
        const g = this.g, W = this.cv.width, H = this.cv.height;
        g.setTransform(1, 0, 0, 1, 0, 0);
        g.fillStyle = "#0a0d17"; g.fillRect(0, 0, W, H);
        if (this.autoFit && t.trace.length >= 4) {
            let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
            const all = t.trace.concat(t.wallCam, t.wallMic);
            for (let i = 0; i + 1 < all.length; i += 2) {
                const ax = all[i], ay = all[i + 1];
                if (ax === 0 && ay === 0) continue;
                x0 = Math.min(x0, ax); x1 = Math.max(x1, ax);
                y0 = Math.min(y0, ay); y1 = Math.max(y1, ay);
            }
            const spanX = Math.max(x1 - x0, 2), spanY = Math.max(y1 - y0, 2);
            const fit = Math.min(W / spanX, H / spanY) * 0.7;
            this.scale = fit / (Math.min(W, H) / 80);
            const midX = (x0 + x1) / 2, midY = (y0 + y1) / 2;
            this.offX = -midX * fit / devicePixelRatio;
            this.offY = midY * fit / devicePixelRatio;
        }
        const cx = W / 2 + this.offX * devicePixelRatio;
        const cy = H / 2 + this.offY * devicePixelRatio;
        const baseScale = Math.min(W, H) / 80 * this.scale;
        const toScreen = (wx, wy) => [cx + wx * baseScale, cy - wy * baseScale];
        // grid: 5m spacing
        const gs = 5 * baseScale;
        if (gs > 5) {
            g.lineWidth = 1;
            let sx = ((cx % gs) + gs) % gs - gs;
            for (let x = sx; x < W + gs; x += gs) {
                g.strokeStyle = Math.abs(x - cx) < gs / 2 && Math.abs(x - cx) > 0 ? "rgba(255,255,255,0.06)" : "rgba(255,255,255,0.06)";
                if (Math.abs(x - cx) < 1) g.strokeStyle = "rgba(255,255,255,0.12)";
                g.beginPath(); g.moveTo(x, 0); g.lineTo(x, H); g.stroke();
            }
            let sy = ((cy % gs) + gs) % gs - gs;
            for (let y = sy; y < H + gs; y += gs) {
                g.strokeStyle = Math.abs(y - cy) < 1 ? "rgba(255,255,255,0.12)" : "rgba(255,255,255,0.06)";
                g.beginPath(); g.moveTo(0, y); g.lineTo(W, y); g.stroke();
            }
        }
        const dots = (data, color, r) => {
            g.fillStyle = color; g.shadowColor = color; g.shadowBlur = 6 * devicePixelRatio;
            for (let i = 0; i + 1 < data.length; i += 2) {
                if (data[i] === 0 && data[i+1] === 0 && i > 0) continue;
                const [x, y] = toScreen(data[i], data[i + 1]);
                g.beginPath(); g.arc(x, y, r * devicePixelRatio, 0, 7); g.fill();
            }
            g.shadowBlur = 0;
        };
        dots(t.wallCam, this.colors.cam, 2.5);
        dots(t.wallMic, this.colors.mic, 2.5);
        dots(t.doors, this.colors.door, 4);
        // trace path
        if (t.trace.length >= 4) {
            g.strokeStyle = this.colors.trace; g.lineWidth = 2 * devicePixelRatio;
            g.shadowColor = this.colors.trace; g.shadowBlur = 8 * devicePixelRatio;
            g.beginPath();
            for (let i = 0; i + 1 < t.trace.length; i += 2) {
                const [x, y] = toScreen(t.trace[i], t.trace[i + 1]);
                i === 0 ? g.moveTo(x, y) : g.lineTo(x, y);
            }
            g.stroke(); g.shadowBlur = 0;
        }
        // origin marker
        const [ox, oy] = toScreen(0, 0);
        g.strokeStyle = "rgba(0,242,255,0.6)"; g.lineWidth = devicePixelRatio;
        g.strokeRect(ox - 4 * devicePixelRatio, oy - 4 * devicePixelRatio, 8 * devicePixelRatio, 8 * devicePixelRatio);
        // current position
        if (t.trace.length >= 2) {
            const [px, py] = toScreen(t.trace[t.trace.length - 2], t.trace[t.trace.length - 1]);
            g.fillStyle = "#ffffff"; g.shadowColor = this.colors.trace; g.shadowBlur = 12 * devicePixelRatio;
            g.beginPath(); g.arc(px, py, 5 * devicePixelRatio, 0, 7); g.fill();
            g.shadowBlur = 0;
            // heading tick
            const yawRad = t.yaw * Math.PI / 180;
            g.strokeStyle = "#ffffff"; g.lineWidth = 2 * devicePixelRatio;
            g.beginPath(); g.moveTo(px, py);
            g.lineTo(px + Math.cos(yawRad) * 16 * devicePixelRatio, py - Math.sin(yawRad) * 16 * devicePixelRatio);
            g.stroke();
        }
        if (t.trace.length === 0) {
            g.fillStyle = "rgba(255,255,255,0.4)";
            g.font = `${14 * devicePixelRatio}px monospace`;
            g.textAlign = "center";
            g.fillText("Start scanning to map", W / 2, H / 2);
        }
    }
}

// ---------- Demo mode: synthetic room, virtual walk ----------
class DemoWalk {
    // 4m x 5m room centered on origin, walls at x=+-2, y=+-2.5; a "door"
    // recess on the right wall between y in [0.5, 1.5] (setback 0.4m).
    constructor(tracker) {
        this.t = tracker; this.x = 0; this.y = -1.5; this.yaw = 90; this.i = 0;
        this.path = this.buildPath();
    }
    buildPath() {
        // rectangle loop inside the room with 90-deg turns, then back to start
        const pts = [];
        const corners = [[0, -1.5], [1.2, -1.5], [1.2, 0.9], [-1.2, 0.9], [-1.2, -1.5], [0, -1.5]];
        for (let c = 0; c < corners.length - 1; c++) {
            const [x0, y0] = corners[c], [x1, y1] = corners[c + 1];
            const legLen = Math.hypot(x1 - x0, y1 - y0);
            // match the tracker's displacement per tick (stepLength / scanRate)
            const stepsPerLeg = Math.max(1, Math.round(legLen / 0.05));
            for (let k = 0; k < stepsPerLeg; k++) {
                const f = k / stepsPerLeg;
                pts.push([x0 + (x1 - x0) * f, y0 + (y1 - y0) * f,
                    Math.atan2(y1 - y0, x1 - x0) * 180 / Math.PI]);
            }
        }
        return pts;
    }
    distToWall(x, y, angDeg) {
        // ray-cast in the synthetic room, returns distance (m)
        const a = angDeg * Math.PI / 180, dx = Math.cos(a), dy = Math.sin(a);
        let best = 5;
        const walls = [[-2, -2.5, 2, -2.5], [2, -2.5, 2, 2.5], [2, 2.5, -2, 2.5], [-2, 2.5, -2, -2.5]];
        for (const [x1, y1, x2, y2] of walls) {
            const den = dx * (y2 - y1) - dy * (x2 - x1);
            if (Math.abs(den) < 1e-9) continue;
            const s = ((x1 - x) * (y2 - y1) - (y1 - y) * (x2 - x1)) / den;
            const u = ((x1 - x) * dy - (y1 - y) * dx) / den;
            if (s > 0.05 && u >= 0 && u <= 1) {
                // door recess on the right wall (x = 2) for y in [0.3, 1.1]
                let d = s;
                if (x1 === 2 && x2 === 2) {
                    const hy = y + s * dy;
                    if (hy > 0.3 && hy < 1.1) d += 0.4;
                }
                best = Math.min(best, d);
            }
        }
        return best;
    }
    tick() {
        if (this.i >= this.path.length) return false;
        const [x, y, heading] = this.path[this.i];
        // tracker moves itself; we only feed yaw + distances + "walking"
        this.yaw = heading;
        const left = this.distToWall(x, y, heading + 90);
        const right = this.distToWall(x, y, heading - 90);
        this.t.update(heading, left, right, 1.0); // accelMag > threshold => walking
        this.i++;
        return true;
    }
}

// ---------- App wiring ----------
(function () {
    const $ = id => document.getElementById(id);
    const tracker = new MotionTracker();
    const map = new MapCanvas($("mapCanvas"));
    let sonar = null, sonarDist = [0, 0];
    let motionYaw = 0, accelMag = 0, hasMotion = false;
    let tickTimer = null, demo = null, demoTimer = null;
    let keys = {};

    const status = m => { $("status").textContent = m; };

    async function requestMotionPermission() {
        try {
            if (typeof DeviceMotionEvent !== "undefined" &&
                typeof DeviceMotionEvent.requestPermission === "function") {
                const r = await DeviceMotionEvent.requestPermission();
                if (r !== "granted") return false;
            }
            if (typeof DeviceOrientationEvent !== "undefined" &&
                typeof DeviceOrientationEvent.requestPermission === "function") {
                const r = await DeviceOrientationEvent.requestPermission();
                if (r !== "granted") return false;
            }
            return true;
        } catch (e) { return false; }
    }

    window.addEventListener("devicemotion", e => {
        const a = e.acceleration || e.accelerationIncludingGravity;
        if (!a) return;
        hasMotion = true;
        accelMag = Math.hypot(a.x || 0, a.y || 0, a.z || 0);
        if (!e.acceleration && e.accelerationIncludingGravity) accelMag = Math.abs(accelMag - 9.81);
    });
    window.addEventListener("deviceorientation", e => {
        if (e.alpha == null) return;
        hasMotion = true;
        // iOS gives a compass heading directly; others give alpha (CCW)
        motionYaw = (e.webkitCompassHeading != null) ? -e.webkitCompassHeading : e.alpha;
    });

    window.addEventListener("keydown", e => { keys[e.key] = true; });
    window.addEventListener("keyup", e => { keys[e.key] = false; });

    async function start() {
        $("btnStart").disabled = true;
        tracker.reset();
        tracker.isTracking = true;

        const motionOk = await requestMotionPermission();
        try {
            sonar = new SonarEngine((l, r) => { sonarDist = [l, r]; });
            await sonar.start();
            status(motionOk || hasMotion
                ? "Scanning. Hold the phone horizontally and walk the room."
                : "Scanning (sonar only — no motion sensors; use arrow keys to walk).");
        } catch (e) {
            status("Microphone unavailable (" + (e.name || e.message) + "). Map still works with arrow keys / demo.");
        }
        $("btnStop").disabled = false;
        $("btnLoop").disabled = false;
        $("btnExport").disabled = false;

        let keyYaw = 0;
        tickTimer = setInterval(() => {
            let yaw = motionYaw, walkingAccel = accelMag;
            if (keys["ArrowLeft"]) keyYaw += 3;
            if (keys["ArrowRight"]) keyYaw -= 3;
            if (!hasMotion) {
                yaw = keyYaw;
                if (keys["ArrowUp"]) walkingAccel = 1.0;
            }
            tracker.update(yaw, sonarDist[0], sonarDist[1], walkingAccel);
            accelMag = 0;
        }, 1000 / tracker.sensorScanRate);
    }

    function stop() {
        if (tickTimer) { clearInterval(tickTimer); tickTimer = null; }
        if (sonar) { sonar.stop(); sonar = null; }
        tracker.isTracking = false;
        sonarDist = [0, 0];
        $("btnStart").disabled = false;
        $("btnStop").disabled = true;
        status("Stopped. Reset to clear the map, or start again.");
    }

    function runDemo() {
        if (demoTimer) { clearInterval(demoTimer); demoTimer = null; $("btnDemo").textContent = "Demo walk (no mic needed)"; return; }
        stop();
        tracker.reset();
        tracker.isTracking = true;
        demo = new DemoWalk(tracker);
        $("btnLoop").disabled = false;
        $("btnExport").disabled = false;
        $("btnDemo").textContent = "Stop demo";
        status("Demo: virtual walk around a 4x5 m room (watch for the door on the right wall).");
        demoTimer = setInterval(() => {
            if (!demo.tick()) {
                clearInterval(demoTimer); demoTimer = null;
                $("btnDemo").textContent = "Demo walk (no mic needed)";
                status("Demo finished. Try 'Close the loop' to see drift correction.");
            }
        }, 50);
    }

    $("btnStart").addEventListener("click", start);
    $("btnStop").addEventListener("click", stop);
    $("btnReset").addEventListener("click", () => { tracker.reset(); map.autoFit = true; status("Map cleared."); });
    $("btnLoop").addEventListener("click", () => {
        tracker.applyLoopClosure();
        status("Loop closed: drift distributed back across the trace.");
    });
    $("btnExport").addEventListener("click", () => {
        const a = document.createElement("a");
        a.download = "batmapper-map.png";
        a.href = $("mapCanvas").toDataURL("image/png");
        a.click();
    });
    $("btnDemo").addEventListener("click", runDemo);

    // UI refresh + render loop
    setInterval(() => {
        $("distL").textContent = sonar && sonar.distL ? sonar.distL.toFixed(2) + " m" : "—";
        $("distR").textContent = sonar && sonar.distR ? sonar.distR.toFixed(2) + " m" : "—";
        $("yaw").textContent = Math.round(tracker.yaw) + "°";
        $("points").textContent = tracker.pointCount;
        $("walking").textContent = tracker.isTracking ? (tracker.isWalking ? "walking" : "idle") : "stopped";
    }, 200);
    (function frame() { map.draw(tracker); requestAnimationFrame(frame); })();
})();
