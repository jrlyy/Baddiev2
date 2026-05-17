import { useState } from "react";

const STRATEGIES = {
  intercept: { color: "#FF6B35", label: "Intercept", icon: "⚡" },
  defensive: { color: "#4ECDC4", label: "Defensive", icon: "🛡" },
  move_to_net: { color: "#FFE66D", label: "Move to Net", icon: "↗" },
  create_depth: { color: "#A855F7", label: "Create Depth", icon: "↕" },
  passive: { color: "#94A3B8", label: "Passive", icon: "○" },
};

const SHOT_TYPES = ["Clear","Drop","Smash","Net shot","Drive","Lift","Push","Lob","Block","Cut"];

function generateMockRally() {
  const strats = Object.keys(STRATEGIES);
  const n = 8 + Math.floor(Math.random() * 12);
  return Array.from({ length: n }, (_, i) => {
    const strategy = strats[Math.floor(Math.random() * strats.length)];
    const confidence = 0.45 + Math.random() * 0.5;
    const margin = Math.random() * 0.6;
    const distances = {};
    strats.forEach(s => { distances[s] = s === strategy ? 1 - confidence : 0.5 + Math.random() * 2; });
    return {
      id: i, shotNumber: i + 1, strategy, confidence, margin, distances,
      lowConfidence: margin < 0.15,
      timestamp: (i * 0.8 + Math.random() * 0.3).toFixed(2),
      player: i % 2 === 0 ? 1 : 2,
      shotType: SHOT_TYPES[Math.floor(Math.random() * SHOT_TYPES.length)],
      shotTypeConf: 0.5 + Math.random() * 0.45,
      shuttle: { x: -3 + Math.random() * 6, y: -6.7 + Math.random() * 13.4 },
      shuttleTraj: Array.from({ length: 16 }, (_, f) => ({
        x: 0.3 + f * 0.025 + (Math.random() - 0.5) * 0.03,
        y: 0.2 + Math.sin(f * 0.3) * 0.15 + (Math.random() - 0.5) * 0.02,
      })),
      tracknetConf: 0.7 + Math.random() * 0.3,
      wristAgrees: Math.random() > 0.2,
      features: {
        velocity: (Math.random() * 3).toFixed(2),
        distToNet: (1 + Math.random() * 6).toFixed(2),
        distToOpponent: (2 + Math.random() * 8).toFixed(2),
        shoulderAngle: (30 + Math.random() * 120).toFixed(1),
        courtPosition: Math.random() > 0.5 ? "Front" : "Rear",
      },
    };
  });
}

function generateEmbeddings(shots) {
  const c = { intercept:{x:.25,y:.3}, defensive:{x:.75,y:.7}, move_to_net:{x:.2,y:.75},
    create_depth:{x:.7,y:.25}, passive:{x:.5,y:.5} };
  return shots.map(s => ({ x: c[s.strategy].x+(Math.random()-.5)*.15,
    y: c[s.strategy].y+(Math.random()-.5)*.15, strategy:s.strategy, shotNumber:s.shotNumber }));
}

function CourtMini({ shots, selectedShot, size = 150 }) {
  const h = size * (13.4 / 6.1); const p = 8;
  const tx = cx => p + ((cx + 3.05) / 6.1) * (size - p * 2);
  const ty = cy => p + ((cy + 6.7) / 13.4) * (h - p * 2);
  return (
    <svg width={size} height={h} style={{ borderRadius: 6, background: "#0a2e0a" }}>
      <rect x={p} y={p} width={size-p*2} height={h-p*2} fill="none" stroke="#2d5a2d" strokeWidth={1.5}/>
      <line x1={p} y1={h/2} x2={size-p} y2={h/2} stroke="#f8fafc44" strokeWidth={1} strokeDasharray="4,3"/>
      <line x1={p} y1={ty(-4.72)} x2={size-p} y2={ty(-4.72)} stroke="#2d5a2d" strokeWidth={.5}/>
      <line x1={p} y1={ty(4.72)} x2={size-p} y2={ty(4.72)} stroke="#2d5a2d" strokeWidth={.5}/>
      <line x1={size/2} y1={ty(-4.72)} x2={size/2} y2={ty(4.72)} stroke="#2d5a2d" strokeWidth={.5}/>
      {shots.map((s,i) => {
        const sel = selectedShot === i;
        return <circle key={i} cx={tx(s.shuttle.x)} cy={ty(s.shuttle.y)} r={sel?5:3}
          fill={STRATEGIES[s.strategy].color} opacity={sel?1:.5}
          stroke={sel?"#f8fafc":"none"} strokeWidth={sel?1.5:0}/>;
      })}
      <text x={size/2} y={h/2-5} textAnchor="middle" fill="#f8fafc33" fontSize={8}
        fontFamily="'JetBrains Mono',monospace">NET</text>
    </svg>
  );
}

function SkeletonViz({ highlight, size = 100 }) {
  const j = { hd:[.5,.08],nk:[.5,.18],lS:[.35,.22],rS:[.65,.22],lE:[.25,.38],rE:[.75,.38],
    lW:[.18,.52],rW:[.82,.52],lH:[.4,.52],rH:[.6,.52],lK:[.38,.72],rK:[.62,.72],lA:[.36,.92],rA:[.64,.92] };
  const b = [["hd","nk"],["nk","lS"],["nk","rS"],["lS","lE"],["rS","rE"],["lE","lW"],["rE","rW"],
    ["nk","lH"],["nk","rH"],["lH","lK"],["rH","rK"],["lK","lA"],["rK","rA"],["lH","rH"]];
  const jt = v => v + (Math.random()-.5)*.04;
  return (
    <svg width={size} height={size} viewBox="0 0 1 1">
      {b.map(([a,c],i) => <line key={i} x1={jt(j[a][0])} y1={jt(j[a][1])} x2={jt(j[c][0])} y2={jt(j[c][1])}
        stroke={highlight||"#64748b"} strokeWidth=".025" strokeLinecap="round" opacity={.7}/>)}
      {Object.entries(j).map(([n,[x,y]]) => <circle key={n} cx={jt(x)} cy={jt(y)}
        r={n==="hd"?.04:.02} fill={highlight||"#e2e8f0"} opacity={.9}/>)}
    </svg>
  );
}

function ConfBar({ value, color, label, isTop }) {
  return (
    <div style={{ display:"flex", alignItems:"center", gap:8, marginBottom:4 }}>
      <span style={{ width:90, fontSize:11, color:isTop?"#f8fafc":"#94a3b8",
        fontFamily:"'JetBrains Mono',monospace", textAlign:"right" }}>{label}</span>
      <div style={{ flex:1, height:14, background:"#1e293b", borderRadius:3, overflow:"hidden" }}>
        <div style={{ width:`${value*100}%`, height:"100%", borderRadius:3,
          background:isTop?`linear-gradient(90deg,${color}cc,${color})`:"#334155",
          transition:"width .6s cubic-bezier(.16,1,.3,1)" }}/></div>
      <span style={{ width:42, fontSize:11, color:isTop?"#f8fafc":"#64748b",
        fontFamily:"'JetBrains Mono',monospace" }}>{(value*100).toFixed(0)}%</span>
    </div>
  );
}

function ShuttleTraj({ traj, color }) {
  if (!traj) return null;
  const w = 200, h = 56, pad = 6;
  const xs = traj.map(p => p.x), ys = traj.map(p => p.y);
  const mnX = Math.min(...xs), mxX = Math.max(...xs), mnY = Math.min(...ys), mxY = Math.max(...ys);
  const sx = v => pad + ((v - mnX) / (mxX - mnX + .001)) * (w - pad * 2);
  const sy = v => pad + ((v - mnY) / (mxY - mnY + .001)) * (h - pad * 2);
  const d = traj.map((p, i) => `${i === 0 ? "M" : "L"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(" ");
  return (
    <svg width={w} height={h} style={{ display: "block" }}>
      <rect width={w} height={h} rx={4} fill="#0a0f1a" />
      <path d={d} fill="none" stroke={color || "#f59e0b"} strokeWidth="1.5" opacity={.8} />
      {traj.map((p, i) => <circle key={i} cx={sx(p.x)} cy={sy(p.y)}
        r={i === 0 ? 3 : i === traj.length - 1 ? 3 : 1.5}
        fill={i === 0 ? "#10b981" : i === traj.length - 1 ? "#ef4444" : `${color || "#f59e0b"}66`} />)}
      <text x={pad} y={h - 2} fontSize={7} fill="#475569" fontFamily="'JetBrains Mono',monospace">shuttle (x,y) · TrackNetV3 · T=16</text>
    </svg>
  );
}

function CourtCalibUI({ onDone }) {
  const [pts, setPts] = useState([]);
  const labels = ["Near-left", "Near-right", "Far-right", "Far-left"];
  const click = (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width * 100).toFixed(1);
    const y = ((e.clientY - r.top) / r.height * 100).toFixed(1);
    const next = [...pts, { x, y }];
    setPts(next);
    if (next.length >= 4) setTimeout(onDone, 500);
  };
  return (
    <div style={{ textAlign: "center" }}>
      <div style={{ fontSize: 13, color: "#94a3b8", marginBottom: 12 }}>
        Click the <strong style={{ color: "#e2e8f0" }}>4 court corners</strong> to compute the homography
        <span style={{ display: "block", fontSize: 11, color: "#64748b", marginTop: 4 }}>
          {pts.length < 4 ? `Next: ${labels[pts.length]} (${pts.length}/4)` : "✓ Calibration complete"}</span>
      </div>
      <div onClick={pts.length < 4 ? click : undefined}
        style={{ width: 480, height: 260, margin: "0 auto", background: "#0a0f1a",
          border: "1px solid #334155", borderRadius: 8, position: "relative",
          cursor: pts.length < 4 ? "crosshair" : "default", overflow: "hidden",
          backgroundImage: "linear-gradient(#1e293b22 1px,transparent 1px),linear-gradient(90deg,#1e293b22 1px,transparent 1px)",
          backgroundSize: "40px 40px" }}>
        <div style={{ position: "absolute", left: "15%", top: "10%", right: "15%", bottom: "10%",
          border: "1px solid #33415588", borderRadius: 2 }}>
          <div style={{ position: "absolute", left: 0, right: 0, top: "50%", height: 1, background: "#33415588" }} />
          <div style={{ position: "absolute", left: "10%", right: "10%", top: "20%", bottom: "20%",
            border: "1px dashed #1e293b55" }} />
        </div>
        {pts.map((p, i) => (
          <div key={i} style={{ position: "absolute", left: `${p.x}%`, top: `${p.y}%`,
            transform: "translate(-50%,-50%)" }}>
            <div style={{ width: 12, height: 12, borderRadius: "50%", background: "#FF6B35",
              border: "2px solid #fff", boxShadow: "0 0 8px #FF6B3588" }} />
            <div style={{ position: "absolute", top: 14, left: "50%", transform: "translateX(-50%)",
              fontSize: 9, color: "#FF6B35", whiteSpace: "nowrap",
              fontFamily: "'JetBrains Mono',monospace" }}>{labels[i]}</div>
          </div>
        ))}
        {pts.length >= 4 && (
          <svg style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%" }}>
            <polygon points={pts.map(p => `${p.x * 4.8},${p.y * 2.6}`).join(" ")}
              fill="#FF6B3511" stroke="#FF6B35" strokeWidth="1.5" strokeDasharray="4,3" />
          </svg>
        )}
      </div>
      <div style={{ marginTop: 8, fontSize: 10, color: "#475569",
        fontFamily: "'JetBrains Mono',monospace" }}>pixel → court (13.4m × 6.1m)</div>
    </div>
  );
}

function EmbedSpace({ points, selectedShot }) {
  const c = { intercept:[25,30], defensive:[75,70], move_to_net:[20,75], create_depth:[70,25], passive:[50,50] };
  return (
    <div style={{ position:"relative", width:"100%", aspectRatio:"1", background:"#0c1222",
      borderRadius:8, overflow:"hidden", border:"1px solid #1e293b" }}>
      <div style={{ position:"absolute", top:8, left:10, fontSize:10, color:"#475569",
        fontFamily:"'JetBrains Mono',monospace", letterSpacing:1 }}>t-SNE EMBEDDING SPACE</div>
      {[.25,.5,.75].map(v => <div key={v}>
        <div style={{ position:"absolute",left:0,right:0,top:`${v*100}%`,height:1,background:"#1e293b44" }}/>
        <div style={{ position:"absolute",top:0,bottom:0,left:`${v*100}%`,width:1,background:"#1e293b44" }}/></div>)}
      {Object.entries(STRATEGIES).map(([k,{icon}]) => {
        const [cx,cy] = c[k];
        return <div key={k} style={{ position:"absolute",left:`${cx}%`,top:`${cy}%`,
          transform:"translate(-50%,-50%)",fontSize:16,opacity:.3 }}>{icon}</div>; })}
      {points.map((pt,i) => {
        const sel = selectedShot!==null && pt.shotNumber===selectedShot+1;
        return <div key={i} style={{ position:"absolute",left:`${pt.x*100}%`,top:`${pt.y*100}%`,
          width:sel?12:7,height:sel?12:7,borderRadius:"50%",background:STRATEGIES[pt.strategy].color,
          border:sel?"2px solid #f8fafc":"1px solid #0f172a",transform:"translate(-50%,-50%)",
          opacity:sel?1:.7,transition:"all .3s",zIndex:sel?10:1,
          boxShadow:sel?`0 0 12px ${STRATEGIES[pt.strategy].color}88`:"none" }}/>; })}
    </div>
  );
}

export default function App() {
  const [stage, setStage] = useState("upload");
  const [shots, setShots] = useState([]);
  const [sel, setSel] = useState(null);
  const [emb, setEmb] = useState([]);
  const [step, setStep] = useState(0);
  const [courtConf, setCourtConf] = useState(0);

  const steps = [
    { label:"Extracting frames", detail:"ffmpeg @ 30fps", icon:"🎞" },
    { label:"Detecting court lines", detail:"Canny → Hough → RANSAC homography", icon:"📐" },
    { label:"Tracking shuttle", detail:"TrackNetV3 (pre-trained, frozen)", icon:"🏸" },
    { label:"Detecting hit events", detail:"Trajectory direction changes (>90°)", icon:"💥" },
    { label:"Pose estimation", detail:"YOLOv8-Pose + Kalman filter", icon:"🦴" },
    { label:"Segmenting shots", detail:"T=16 frames centered on each hit", icon:"✂️" },
    { label:"Feature engineering", detail:"L0–L3 court-relative node features", icon:"🔧" },
    { label:"Encoding motions", detail:"ST-GCN → 256-dim embeddings", icon:"🧠" },
    { label:"Classifying strategies", detail:"ProtoNet distance + aux shot-type head", icon:"🎯" },
  ];

  const go = () => {
    setStage("processing"); setStep(0);
    let s = 0;
    const iv = setInterval(() => {
      s++; setStep(s);
      if (s >= steps.length) { clearInterval(iv);
        setTimeout(() => {
          const r = generateMockRally();
          setShots(r); setEmb(generateEmbeddings(r));
          setCourtConf(.85+Math.random()*.14);
          setStage("results"); setSel(0);
        }, 400); }
    }, 550);
  };

  const d = sel!==null ? shots[sel] : null;
  const sc = {}; shots.forEach(s => { sc[s.strategy]=(sc[s.strategy]||0)+1; });
  const dom = Object.entries(sc).sort((a,b) => b[1]-a[1])[0]?.[0];
  const avgC = shots.length? shots.reduce((a,s) => a+s.confidence,0)/shots.length : 0;
  const flagged = shots.filter(s => s.lowConfidence).length;
  const M = "'JetBrains Mono',monospace";

  return (
    <div style={{ minHeight:"100vh", background:"#0a0f1a", color:"#e2e8f0",
      fontFamily:"'DM Sans','Helvetica Neue',sans-serif" }}>
      <link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@300;400;500&family=Instrument+Serif:ital@0;1&display=swap" rel="stylesheet"/>

      {/* Header */}
      <div style={{ borderBottom:"1px solid #1e293b", padding:"14px 24px",
        display:"flex", alignItems:"center", justifyContent:"space-between" }}>
        <div style={{ display:"flex", alignItems:"center", gap:12 }}>
          <div style={{ width:32,height:32,borderRadius:6,
            background:"linear-gradient(135deg,#FF6B35,#A855F7)",
            display:"flex",alignItems:"center",justifyContent:"center",fontSize:16 }}>🏸</div>
          <div>
            <div style={{ fontSize:15, fontWeight:600, letterSpacing:"-.02em" }}>Tactical Strategy Analyzer</div>
            <div style={{ fontSize:10, color:"#64748b", fontFamily:M, letterSpacing:.5 }}>
              COURT DETECT → TRACKNET → SKELETON → ST-GCN → PROTONET</div>
          </div>
        </div>
        {stage==="results" && <button onClick={() => { setStage("upload"); setShots([]); setSel(null); }}
          style={{ background:"#1e293b",border:"1px solid #334155",color:"#94a3b8",
            padding:"6px 14px",borderRadius:6,fontSize:12,cursor:"pointer",fontFamily:M }}>New Analysis</button>}
      </div>

      {/* Upload */}
      {stage==="upload" && (
        <div style={{ display:"flex",alignItems:"center",justifyContent:"center",
          minHeight:"calc(100vh - 61px)",padding:24 }}>
          <div style={{ textAlign:"center", maxWidth:520 }}>
            <div onClick={() => setStage("calibration")} style={{ border:"2px dashed #334155",borderRadius:16,
              padding:"56px 48px",cursor:"pointer",background:"#0f172a",transition:"border-color .3s" }}
              onMouseEnter={e => e.currentTarget.style.borderColor="#FF6B35"}
              onMouseLeave={e => e.currentTarget.style.borderColor="#334155"}>
              <div style={{ fontSize:48,marginBottom:16,opacity:.6 }}>🎬</div>
              <div style={{ fontSize:22,fontWeight:600,fontFamily:"'Instrument Serif',serif",
                fontStyle:"italic",marginBottom:8 }}>Upload match footage</div>
              <div style={{ fontSize:13,color:"#64748b",lineHeight:1.7 }}>
                Drop a badminton rally video (.mp4, .avi, .mov)<br/>
                Court calibration, shuttle tracking, skeleton extraction,<br/>
                and tactical strategy classification per shot.</div>
              <div style={{ marginTop:24,display:"inline-block",
                background:"linear-gradient(135deg,#FF6B35,#A855F7)",color:"#fff",
                padding:"10px 28px",borderRadius:8,fontSize:13,fontWeight:600 }}>Select Video</div>
            </div>
            <div style={{ marginTop:20,fontSize:10,color:"#475569",fontFamily:M,lineHeight:2 }}>
              Phase C: Calibrate → ffmpeg → TrackNetV3 (shots + shuttle) → YOLOv8-Pose<br/>
              → Segment → Features (L0–L3) → ST-GCN → ProtoNet + Aux shot-type head
            </div>
          </div>
        </div>
      )}

      {/* Court calibration */}
      {stage==="calibration" && (
        <div style={{ display:"flex",alignItems:"center",justifyContent:"center",
          minHeight:"calc(100vh - 61px)",padding:24 }}>
          <div style={{ maxWidth:560 }}>
            <div style={{ textAlign:"center",marginBottom:24 }}>
              <div style={{ fontSize:10,color:"#64748b",fontFamily:M,letterSpacing:1,marginBottom:6 }}>
                PHASE C · STEP 1</div>
              <div style={{ fontSize:22,fontFamily:"'Instrument Serif',serif",fontStyle:"italic" }}>
                Court Calibration</div>
            </div>
            <CourtCalibUI onDone={go}/>
            <div style={{ textAlign:"center",marginTop:16 }}>
              <button onClick={go} style={{ background:"transparent",border:"1px solid #334155",
                color:"#64748b",padding:"6px 16px",borderRadius:6,fontSize:11,cursor:"pointer",
                fontFamily:M }}>Skip (use auto-detection)</button>
            </div>
          </div>
        </div>
      )}

      {/* Processing */}
      {stage==="processing" && (
        <div style={{ display:"flex",alignItems:"center",justifyContent:"center",
          minHeight:"calc(100vh - 61px)",padding:24 }}>
          <div style={{ width:460 }}>
            <div style={{ textAlign:"center",marginBottom:28,fontFamily:"'Instrument Serif',serif",
              fontSize:22,fontStyle:"italic" }}>Analyzing rally...</div>
            {steps.map((s,i) => {
              const done = i<step, act = i===step;
              return (
                <div key={i} style={{ display:"flex",alignItems:"center",gap:12,
                  padding:"8px 0",borderBottom:"1px solid #1e293b22",
                  opacity:done?1:act?1:.3,transition:"opacity .4s" }}>
                  <div style={{ width:28,height:28,borderRadius:"50%",display:"flex",
                    alignItems:"center",justifyContent:"center",fontSize:13,
                    background:done?"#10b981":act?"#FF6B35":"#1e293b",
                    color:done||act?"#fff":"#475569",transition:"all .4s" }}>
                    {done?"✓":s.icon}</div>
                  <div style={{ flex:1 }}>
                    <div style={{ fontSize:13,fontWeight:500 }}>{s.label}</div>
                    <div style={{ fontSize:10,color:"#64748b",fontFamily:M }}>{s.detail}</div>
                  </div>
                  {act && <div style={{ width:16,height:16,border:"2px solid #FF6B35",
                    borderTopColor:"transparent",borderRadius:"50%",animation:"spin .8s linear infinite" }}/>}
                </div>
              );
            })}
            <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
          </div>
        </div>
      )}

      {/* Results */}
      {stage==="results" && (
        <div style={{ display:"flex",height:"calc(100vh - 61px)" }}>
          {/* Left */}
          <div style={{ width:360,borderRight:"1px solid #1e293b",display:"flex",
            flexDirection:"column",flexShrink:0 }}>
            <div style={{ padding:"12px 14px 10px",borderBottom:"1px solid #1e293b" }}>
              <div style={{ fontSize:10,color:"#64748b",fontFamily:M,letterSpacing:1,marginBottom:8 }}>RALLY SUMMARY</div>
              <div style={{ display:"grid",gridTemplateColumns:"1fr 1fr 1fr 1fr",gap:6 }}>
                {[
                  {v:shots.length,l:"shots",c:"#e2e8f0"},
                  {v:`${(avgC*100).toFixed(0)}%`,l:"avg conf",c:avgC>.7?"#10b981":"#f59e0b"},
                  {v:flagged,l:"flagged",c:flagged>0?"#ef4444":"#10b981"},
                  {v:`${(courtConf*100).toFixed(0)}%`,l:"court det.",c:courtConf>.9?"#10b981":"#f59e0b"},
                ].map(({v,l,c},i) => (
                  <div key={i} style={{ background:"#0f172a",borderRadius:6,padding:"7px 8px",
                    border:"1px solid #1e293b" }}>
                    <div style={{ fontSize:17,fontWeight:700,fontFamily:M,color:c }}>{v}</div>
                    <div style={{ fontSize:8,color:"#64748b" }}>{l}</div>
                  </div>
                ))}
              </div>
              <div style={{ display:"flex",height:5,borderRadius:3,overflow:"hidden",marginTop:8 }}>
                {Object.entries(sc).sort((a,b)=>b[1]-a[1]).map(([s,c]) =>
                  <div key={s} style={{ width:`${(c/shots.length)*100}%`,background:STRATEGIES[s].color }}/>)}
              </div>
            </div>
            <div style={{ flex:1,overflowY:"auto" }}>
              {shots.map((s,i) => (
                <div key={i} onClick={() => setSel(i)} style={{ display:"flex",alignItems:"center",
                  gap:8,padding:"8px 14px",cursor:"pointer",
                  background:sel===i?"#1e293b":"transparent",
                  borderLeft:sel===i?`3px solid ${STRATEGIES[s.strategy].color}`:"3px solid transparent",
                  borderBottom:"1px solid #1e293b22",transition:"all .15s" }}>
                  <div style={{ width:26,height:26,borderRadius:5,
                    background:`${STRATEGIES[s.strategy].color}22`,display:"flex",
                    alignItems:"center",justifyContent:"center",fontSize:13,flexShrink:0 }}>
                    {STRATEGIES[s.strategy].icon}</div>
                  <div style={{ flex:1,minWidth:0 }}>
                    <div style={{ display:"flex",alignItems:"center",gap:5,flexWrap:"wrap" }}>
                      <span style={{ fontSize:11,fontWeight:500 }}>Shot {s.shotNumber}</span>
                      <span style={{ fontSize:10,color:STRATEGIES[s.strategy].color,fontFamily:M }}>
                        {STRATEGIES[s.strategy].label}</span>
                      <span style={{ fontSize:9,color:"#64748b",fontFamily:M,
                        background:"#1e293b",padding:"1px 5px",borderRadius:3 }}>{s.shotType}</span>
                      {s.lowConfidence && <span style={{ fontSize:8,background:"#ef444433",
                        color:"#fca5a5",padding:"1px 5px",borderRadius:3,fontFamily:M }}>LOW</span>}
                    </div>
                    <div style={{ display:"flex",alignItems:"center",gap:4,marginTop:2 }}>
                      <div style={{ flex:1,height:3,background:"#1e293b",borderRadius:2,overflow:"hidden" }}>
                        <div style={{ width:`${s.confidence*100}%`,height:"100%",
                          background:STRATEGIES[s.strategy].color,borderRadius:2 }}/></div>
                      <span style={{ fontSize:10,color:"#64748b",fontFamily:M,width:28,textAlign:"right" }}>
                        {(s.confidence*100).toFixed(0)}%</span>
                    </div>
                  </div>
                  <span style={{ fontSize:9,color:"#475569",fontFamily:M }}>P{s.player}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Right */}
          <div style={{ flex:1,overflowY:"auto",padding:22 }}>
            {d && (<div>
              <div style={{ display:"flex",alignItems:"flex-start",justifyContent:"space-between",marginBottom:18 }}>
                <div>
                  <div style={{ fontSize:10,color:"#64748b",fontFamily:M,letterSpacing:1,marginBottom:4 }}>
                    SHOT {d.shotNumber} OF {shots.length} · PLAYER {d.player} · {d.timestamp}s</div>
                  <div style={{ display:"flex",alignItems:"center",gap:10 }}>
                    <span style={{ fontSize:30,fontFamily:"'Instrument Serif',serif",fontStyle:"italic",
                      color:STRATEGIES[d.strategy].color }}>{STRATEGIES[d.strategy].label}</span>
                    <span style={{ fontSize:26 }}>{STRATEGIES[d.strategy].icon}</span>
                  </div>
                  <div style={{ display:"flex",alignItems:"center",gap:8,marginTop:6 }}>
                    <div style={{ background:"#1e293b",border:"1px solid #334155",borderRadius:6,
                      padding:"4px 10px",fontSize:12,fontFamily:M }}>
                      <span style={{ color:"#94a3b8" }}>Shot type: </span>
                      <span style={{ color:"#f8fafc",fontWeight:500 }}>{d.shotType}</span>
                      <span style={{ color:"#64748b",marginLeft:6 }}>({(d.shotTypeConf*100).toFixed(0)}%)</span>
                    </div>
                    <div style={{ fontSize:10,color:"#475569",fontFamily:M }}>via aux head (Phase A)</div>
                  </div>
                  <div style={{ display:"flex",alignItems:"center",gap:6,marginTop:6 }}>
                    <div style={{ width:8,height:8,borderRadius:"50%",
                      background:d.wristAgrees?"#10b981":"#f59e0b" }}/>
                    <span style={{ fontSize:10,color:"#64748b",fontFamily:M }}>
                      Hit: TrackNet {(d.tracknetConf*100).toFixed(0)}%
                      {d.wristAgrees?" · wrist ✓":" · wrist ✗"}</span>
                  </div>
                </div>
                {d.lowConfidence && <div style={{ background:"#ef444422",border:"1px solid #ef444444",
                  borderRadius:8,padding:"8px 14px",fontSize:11,color:"#fca5a5",maxWidth:200,lineHeight:1.5 }}>
                  ⚠ Low margin ({(d.margin*100).toFixed(0)}%). Flag for human review.</div>}
              </div>

              <div style={{ display:"grid",gridTemplateColumns:"1fr 1fr",gap:16 }}>
                {/* Prototype distances */}
                <div style={{ background:"#0f172a",borderRadius:10,padding:16,border:"1px solid #1e293b" }}>
                  <div style={{ fontSize:10,color:"#64748b",fontFamily:M,letterSpacing:1,marginBottom:10 }}>
                    PROTOTYPE DISTANCES</div>
                  {Object.entries(STRATEGIES).map(([k,{color,label}]) => {
                    const mx = Math.max(...Object.values(d.distances));
                    return <ConfBar key={k} value={Math.max(0,1-d.distances[k]/(mx+.01))}
                      color={color} label={label} isTop={k===d.strategy}/>;
                  })}
                  <div style={{ marginTop:8,display:"flex",justifyContent:"space-between",fontSize:10,fontFamily:M }}>
                    <span style={{ color:"#64748b" }}>Conf: <span style={{ color:"#e2e8f0" }}>
                      {(d.confidence*100).toFixed(1)}%</span></span>
                    <span style={{ color:"#64748b" }}>Margin: <span style={{
                      color:d.margin<.15?"#ef4444":"#10b981" }}>{(d.margin*100).toFixed(1)}%</span></span>
                  </div>
                </div>

                {/* Skeleton + features */}
                <div style={{ background:"#0f172a",borderRadius:10,padding:16,border:"1px solid #1e293b" }}>
                  <div style={{ fontSize:10,color:"#64748b",fontFamily:M,letterSpacing:1,marginBottom:10 }}>
                    SKELETON · PLAYER {d.player}</div>
                  <div style={{ display:"flex",gap:12,alignItems:"flex-start" }}>
                    <div style={{ background:"#0a0f1a",borderRadius:6,padding:4,border:"1px solid #1e293b" }}>
                      <SkeletonViz highlight={STRATEGIES[d.strategy].color}/></div>
                    <div style={{ flex:1 }}>
                      <div style={{ fontSize:10,color:"#64748b",fontFamily:M,letterSpacing:1,marginBottom:6 }}>
                        NODE FEATURES (L1–L2)</div>
                      {[["Velocity",`${d.features.velocity} m/s`],["Dist to net",`${d.features.distToNet} m`],
                        ["Dist to opp.",`${d.features.distToOpponent} m`],["Shoulder ∠",`${d.features.shoulderAngle}°`],
                        ["Court pos.",d.features.courtPosition]].map(([l,v]) =>
                        <div key={l} style={{ display:"flex",justifyContent:"space-between",padding:"3px 0",
                          borderBottom:"1px solid #1e293b33",fontSize:11 }}>
                          <span style={{ color:"#64748b" }}>{l}</span>
                          <span style={{ fontFamily:M,color:"#e2e8f0" }}>{v}</span></div>)}
                    </div>
                  </div>
                </div>

                {/* Embedding */}
                <div style={{ background:"#0f172a",borderRadius:10,padding:16,border:"1px solid #1e293b" }}>
                  <EmbedSpace points={emb} selectedShot={sel}/>
                  <div style={{ display:"flex",flexWrap:"wrap",gap:8,marginTop:8 }}>
                    {Object.entries(STRATEGIES).map(([k,{color,label}]) =>
                      <div key={k} style={{ display:"flex",alignItems:"center",gap:4,fontSize:10,color:"#94a3b8" }}>
                        <div style={{ width:7,height:7,borderRadius:"50%",background:color }}/>{label}</div>)}
                  </div>
                </div>

                {/* Court + shuttle traj + flow */}
                <div style={{ background:"#0f172a",borderRadius:10,padding:16,border:"1px solid #1e293b",
                  display:"flex",flexDirection:"column",gap:12 }}>
                  <div>
                    <div style={{ fontSize:10,color:"#64748b",fontFamily:M,letterSpacing:1,marginBottom:6 }}>
                      SHUTTLE TRAJECTORY (T=16 FRAMES)</div>
                    <ShuttleTraj traj={d.shuttleTraj} color={STRATEGIES[d.strategy].color}/>
                  </div>
                  <div>
                    <div style={{ display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:6 }}>
                      <div style={{ fontSize:10,color:"#64748b",fontFamily:M,letterSpacing:1 }}>
                        SHUTTLE POSITIONS (COURT VIEW)</div>
                      <div style={{ fontSize:9,fontFamily:M,color:courtConf>.9?"#10b981":"#f59e0b" }}>
                        court det. {(courtConf*100).toFixed(0)}%</div>
                    </div>
                    <div style={{ display:"flex",justifyContent:"center" }}>
                      <CourtMini shots={shots} selectedShot={sel}/></div>
                  </div>
                  <div>
                    <div style={{ fontSize:10,color:"#64748b",fontFamily:M,letterSpacing:1,marginBottom:6 }}>
                      RALLY STRATEGY FLOW</div>
                    <div style={{ display:"flex",gap:2,alignItems:"flex-end",height:46 }}>
                      {shots.map((s,i) => <div key={i} onClick={() => setSel(i)}
                        style={{ flex:1,height:`${s.confidence*100}%`,minHeight:6,
                          background:sel===i?STRATEGIES[s.strategy].color:`${STRATEGIES[s.strategy].color}88`,
                          borderRadius:"2px 2px 0 0",cursor:"pointer",transition:"all .2s" }}
                        title={`Shot ${s.shotNumber}: ${STRATEGIES[s.strategy].label} (${s.shotType})`}/>)}
                    </div>
                    <div style={{ display:"flex",justifyContent:"space-between",marginTop:4,
                      fontSize:8,color:"#475569",fontFamily:M }}>
                      <span>1</span><span>Rally →</span><span>{shots.length}</span></div>
                    <div style={{ marginTop:6,fontSize:10,color:"#64748b" }}>
                      Dominant: <span style={{ color:STRATEGIES[dom]?.color,fontWeight:600 }}>
                        {STRATEGIES[dom]?.label}</span> ({sc[dom]}/{shots.length})</div>
                  </div>
                </div>
              </div>
            </div>)}
          </div>
        </div>
      )}
    </div>
  );
}
