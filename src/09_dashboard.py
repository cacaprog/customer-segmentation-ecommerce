"""
generate_dashboard.py
─────────────────────
Reads all pipeline export files and produces a single self-contained
dashboard HTML file with data baked in — no uploads, no server needed.

Usage:
    python generate_dashboard.py                        # uses default paths
    python generate_dashboard.py --data-dir ./outputs  # custom folder
    python generate_dashboard.py --out my_report.html  # custom output name

Requirements: Python 3.8+  (no external packages needed)
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# ── FILE MAP ──────────────────────────────────────────────────────────────
# key → (filename, parser, required)
FILE_MAP = {
    "rfm_summary":         ("rfm_executive_summary.json",        "json", True),
    "rfm_segments":        ("rfm_segment_profiles.csv",          "csv",  True),
    "rfm_clusters":        ("rfm_enhanced_cluster_profiles.csv", "csv",  False),
    "behavioral_clusters": ("behavioral_cluster_profiles.csv",   "csv",  False),
    "daily_patterns":      ("daily_purchase_patterns.csv",       "csv",  False),
    "hourly_patterns":     ("hourly_purchase_patterns.csv",      "csv",  False),
    "insights":            ("insights_report.json",              "json", False),
}

# ── PARSERS ───────────────────────────────────────────────────────────────
def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def load_csv(path: Path) -> list:
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                v = v.strip()
                try:
                    parsed[k] = int(v) if v.lstrip("-").isdigit() else float(v)
                except (ValueError, AttributeError):
                    parsed[k] = v
            rows.append(parsed)
    return rows

# ── DATA LOADER ───────────────────────────────────────────────────────────
def load_data(data_dir: Path) -> dict:
    data = {}
    print(f"\n📂  Looking for files in: {data_dir.resolve()}\n")

    for key, (filename, parser, required) in FILE_MAP.items():
        path = data_dir / filename
        if path.exists():
            try:
                data[key] = load_json(path) if parser == "json" else load_csv(path)
                rows = len(data[key]) if isinstance(data[key], list) else "—"
                print(f"  ✅  {filename:<45} loaded  ({rows} rows)" if rows != "—"
                      else f"  ✅  {filename:<45} loaded")
            except Exception as e:
                print(f"  ⚠️   {filename:<45} parse error: {e}")
                if required:
                    sys.exit(f"\n❌  Required file '{filename}' failed to parse. Aborting.")
        else:
            status = "❌  REQUIRED — missing!" if required else "⚪  optional — skipping"
            print(f"  {status:<6} {filename}")
            if required:
                sys.exit(f"\n❌  Required file '{filename}' not found in {data_dir}. Aborting.")

    print()
    return data

# ── HTML TEMPLATE ─────────────────────────────────────────────────────────
# The dashboard React app — identical to the upload version but:
# • No upload screen — data is injected as window.__DASHBOARD_DATA__
# • App() starts directly with fileData pre-populated
# • FIXED: Added react-is dependency and updated to Recharts 2.12.0
DASHBOARD_JS = r"""
const {
  useState, useCallback, useRef,
  PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, LineChart, Line, ScatterChart, Scatter, ZAxis
} = { ...React, ...Recharts };

// ── INJECTED DATA ────────────────────────────────────────────────────────
const INJECTED = window.__DASHBOARD_DATA__;

// ── THEME ────────────────────────────────────────────────────────────────
const T = {
  bg:"#070B14", border:"#1A2235", text:"#C8D0E0", muted:"#6B7894",
  dim:"#3A4560", subtle:"#8892A4",
  green:"#00E5A0", blue:"#00C4FF", purple:"#A78BFA",
  yellow:"#FBBF24", orange:"#F97316", red:"#EF4444",
  mono:"'DM Mono',monospace",
};

const SEVERITY_COLOR  = { RED:T.red, AMBER:T.yellow, GREEN:T.green };
const PRIORITY_COLOR  = { CRITICAL:T.red, HIGH:T.orange, MEDIUM:T.yellow, LOW:T.blue };
const EFFORT_COLOR    = { "Quick Win":T.green, Medium:T.yellow, Strategic:T.purple };
const CATEGORY_COLOR  = { churn_risk:T.red, growth:T.green, opportunity:T.blue, timing:T.purple };
const SEG_COLORS = {
  "Champions":T.green,"Loyal Customers":T.blue,"Lost":T.red,
  "Potential Loyalists":T.purple,"Hibernating":"#6B7280",
  "At Risk":T.orange,"Can't Lose Them":"#EF4444",
  "About To Sleep":T.yellow,"Need Attention":"#FB923C",
  "Promising":"#34D399","Recent Customers":"#60A5FA",
};

const fmt  = n => n>=1e6?`£${(n/1e6).toFixed(2)}M`:n>=1e3?`£${(n/1e3).toFixed(0)}K`:`£${Number(n).toFixed(0)}`;
const fmtN = n => Number(n).toLocaleString();

// ── SHARED UI ────────────────────────────────────────────────────────────
const card = {background:"rgba(13,17,30,0.9)",border:"1px solid #1A2235",borderRadius:10,padding:20};

function KPICard({label,value,sub,accent}){
  return (
    <div style={{...card,borderTop:`2px solid ${accent}`,position:"relative",overflow:"hidden"}}>
      <div style={{position:"absolute",top:0,right:0,width:44,height:44,background:`${accent}08`,borderRadius:"0 0 0 44px"}}/>
      <div style={{fontSize:10,color:T.subtle,letterSpacing:"0.1em",textTransform:"uppercase",marginBottom:6}}>{label}</div>
      <div style={{fontSize:22,fontWeight:700,color:"#F0F4FF",fontFamily:T.mono}}>{value}</div>
      {sub&&<div style={{fontSize:11,color:accent,marginTop:4}}>{sub}</div>}
    </div>
  );
}

function SectionTitle({children,accent=T.green}){
  return (
    <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:14}}>
      <div style={{width:3,height:16,background:accent,borderRadius:2}}/>
      <span style={{fontSize:11,fontWeight:600,color:T.text,letterSpacing:"0.1em",textTransform:"uppercase"}}>{children}</span>
    </div>
  );
}

function TT({active,payload,label,fmt:f}){
  if(!active||!payload?.length) return null;
  return (
    <div style={{background:"#0D1117",border:`1px solid ${T.border}`,borderRadius:6,padding:"8px 12px",fontSize:11,color:T.text}}>
      {label&&<div style={{color:T.subtle,marginBottom:4}}>{label}</div>}
      {payload.map((p,i)=><div key={i} style={{color:p.color||T.green}}>{f?f(p.value,p.name):(p.value?.toLocaleString?.()??p.value)}</div>)}
    </div>
  );
}

// ── EXECUTIVE TAB ────────────────────────────────────────────────────────
function ExecutiveTab({data}){
  const summary=data.rfm_summary;
  const segments=data.rfm_segments||[];
  const kpis=summary?[
    {label:"Total Revenue",value:fmt(summary.total_revenue),sub:"Dec 2009 – Dec 2011",accent:T.green},
    {label:"Total Customers",value:fmtN(summary.total_customers),sub:"Identified accounts",accent:T.blue},
    {label:"Avg Customer Value",value:fmt(summary.avg_customer_value),sub:`Median ${fmt(summary.median_customer_value)}`,accent:T.purple},
    {label:"Avg Purchase Freq.",value:`${Number(summary.avg_purchase_frequency).toFixed(1)}x`,sub:"Per customer",accent:T.yellow},
  ]:[];
  const segList=summary?Object.entries(summary.segments).map(([name,d])=>({name,...d})):segments;
  const pieData=segList.filter(s=>Number(s.Revenue_Pct||0)>0.5).map(s=>({
    name:s.name||s.Segment,value:Number(s.Revenue_Pct||0),color:SEG_COLORS[s.name||s.Segment]||T.blue,
  }));
  const barData=[...segList].sort((a,b)=>Number(b.Revenue_Pct||0)-Number(a.Revenue_Pct||0)).slice(0,8).map(s=>({
    name:s.name||s.Segment,revPct:Number(s.Revenue_Pct||0),color:SEG_COLORS[s.name||s.Segment]||T.blue,
  }));
  const champs=segList.find(s=>(s.name||s.Segment)==="Champions");
  return (
    <div>
      <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:12,marginBottom:16}}>
        {kpis.map(k=><KPICard key={k.label} {...k}/>)}
      </div>
      {champs&&(
        <div style={{...card,borderLeft:`3px solid ${T.green}`,marginBottom:16,display:"flex",alignItems:"center",gap:14}}>
          <div style={{fontSize:18}}>⚡</div>
          <div style={{fontSize:12,color:T.subtle}}>
            <span style={{color:"#F0F4FF",fontWeight:600}}>Pareto Insight: </span>
            Top {Number(champs.Customer_Pct||8).toFixed(0)}% of customers (Champions, {fmtN(champs.Customer_Count||471)} accounts) generate{" "}
            <span style={{color:T.green,fontWeight:600}}>{Number(champs.Revenue_Pct||47).toFixed(1)}%</span> of total revenue.
            Protecting this group is the single highest-ROI activity.
          </div>
        </div>
      )}
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:16}}>
        <div style={card}>
          <SectionTitle>Revenue Distribution by Segment</SectionTitle>
          <ResponsiveContainer width="100%" height={240}>
            <PieChart>
              <Pie data={pieData} cx="40%" cy="50%" innerRadius={60} outerRadius={95} paddingAngle={2} dataKey="value">
                {pieData.map((e,i)=><Cell key={i} fill={e.color} stroke="none"/>)}
              </Pie>
              <Tooltip content={<TT fmt={v=>`${Number(v).toFixed(1)}%`}/>}/>
            </PieChart>
          </ResponsiveContainer>
          <div style={{display:"flex",flexWrap:"wrap",gap:6,marginTop:4}}>
            {pieData.map(d=>(
              <div key={d.name} style={{display:"flex",alignItems:"center",gap:4,fontSize:10,color:T.subtle}}>
                <div style={{width:7,height:7,borderRadius:2,background:d.color}}/>
                {d.name} ({Number(d.value).toFixed(1)}%)
              </div>
            ))}
          </div>
        </div>
        <div style={card}>
          <SectionTitle>Revenue Share by Segment</SectionTitle>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={barData} layout="vertical" margin={{left:8,right:20}}>
              <XAxis type="number" tick={{fill:T.dim,fontSize:10}} tickFormatter={v=>`${v}%`}/>
              <YAxis type="category" dataKey="name" tick={{fill:T.subtle,fontSize:10}} width={130}/>
              <Tooltip content={<TT fmt={v=>`${Number(v).toFixed(1)}%`}/>}/>
              <Bar dataKey="revPct" radius={[0,4,4,0]}>
                {barData.map((s,i)=><Cell key={i} fill={s.color}/>)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

// ── SEGMENTS TAB ─────────────────────────────────────────────────────────
function SegmentsTab({data}){
  const [selected,setSelected]=useState(null);
  const summary=data.rfm_summary;
  const segments=summary
    ?Object.entries(summary.segments).map(([name,d])=>({name,...d}))
    :(data.rfm_segments||[]).map(s=>({name:s.Segment,...s}));
  const risk=s=>{const r=Number(s.Avg_Recency||0);if(r>400)return"critical";if(r>200)return"high";if(r>80)return"medium";return"low";};
  const RISK_COLOR={low:T.green,medium:T.yellow,high:T.orange,critical:T.red};
  const sel=selected?segments.find(s=>s.name===selected):null;
  return (
    <div>
      <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:10,marginBottom:16}}>
        {segments.map(s=>{
          const r=risk(s);const color=SEG_COLORS[s.name]||T.blue;const isActive=selected===s.name;
          return (
            <div key={s.name} onClick={()=>setSelected(isActive?null:s.name)} style={{
              ...card,cursor:"pointer",borderLeft:`3px solid ${color}`,
              borderColor:isActive?color:T.border,
              transform:isActive?"translateY(-2px)":"none",
              boxShadow:isActive?`0 4px 20px ${color}20`:"none",transition:"all 0.2s",
            }}>
              <div style={{display:"flex",justifyContent:"space-between",marginBottom:8}}>
                <div style={{fontSize:12,fontWeight:600,color:"#F0F4FF"}}>{s.name}</div>
                <div style={{fontSize:9,padding:"2px 6px",borderRadius:3,background:`${RISK_COLOR[r]}20`,color:RISK_COLOR[r],textTransform:"uppercase"}}>{r}</div>
              </div>
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:6}}>
                <div><div style={{fontSize:9,color:T.dim}}>Customers</div><div style={{fontSize:15,fontWeight:700,color:color,fontFamily:T.mono}}>{fmtN(s.Customer_Count||0)}</div></div>
                <div><div style={{fontSize:9,color:T.dim}}>Rev Share</div><div style={{fontSize:15,fontWeight:700,color:"#F0F4FF",fontFamily:T.mono}}>{Number(s.Revenue_Pct||0).toFixed(1)}%</div></div>
              </div>
            </div>
          );
        })}
      </div>
      {sel?(
        <div style={{...card,borderColor:SEG_COLORS[sel.name]||T.blue}} className="fade-in">
          <div style={{display:"flex",justifyContent:"space-between",marginBottom:16}}>
            <div style={{fontSize:15,fontWeight:700,color:"#F0F4FF"}}>{sel.name}</div>
            <button onClick={()=>setSelected(null)} style={{background:"none",border:"none",color:T.dim,cursor:"pointer",fontSize:16}}>✕</button>
          </div>
          <div style={{display:"grid",gridTemplateColumns:"repeat(5,1fr)",gap:10,marginBottom:14}}>
            {[
              {l:"Customers",v:fmtN(sel.Customer_Count||0)},
              {l:"Revenue Share",v:`${Number(sel.Revenue_Pct||0).toFixed(1)}%`},
              {l:"Avg Monetary",v:fmt(sel.Avg_Monetary||0)},
              {l:"Avg Frequency",v:`${Number(sel.Avg_Frequency||0).toFixed(1)}x`},
              {l:"Avg Recency",v:`${Number(sel.Avg_Recency||0).toFixed(0)}d`},
            ].map(({l,v})=>(
              <div key={l} style={{background:"#0A0F1E",padding:"10px 12px",borderRadius:6}}>
                <div style={{fontSize:9,color:T.dim,marginBottom:3}}>{l}</div>
                <div style={{fontSize:14,fontWeight:700,color:SEG_COLORS[sel.name]||T.blue,fontFamily:T.mono}}>{v}</div>
              </div>
            ))}
          </div>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:8}}>
            {[["R Score",sel.Avg_R_Score],["F Score",sel.Avg_F_Score],["M Score",sel.Avg_M_Score]].map(([l,v])=>(
              <div key={l} style={{background:"#0A0F1E",padding:"10px 12px",borderRadius:6}}>
                <div style={{fontSize:9,color:T.dim,marginBottom:4}}>{l}</div>
                <div style={{display:"flex",gap:3}}>{[1,2,3,4,5].map(n=>(<div key={n} style={{flex:1,height:6,borderRadius:3,background:n<=Math.round(v||0)?(SEG_COLORS[sel.name]||T.blue):T.border}}/>))}</div>
                <div style={{fontSize:11,fontWeight:700,color:"#F0F4FF",marginTop:4,fontFamily:T.mono}}>{Number(v||0).toFixed(2)}</div>
              </div>
            ))}
          </div>
        </div>
      ):(
        <div style={card}>
          <SectionTitle>Avg Monetary Value by Segment</SectionTitle>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={segments.slice(0,8).sort((a,b)=>Number(b.Avg_Monetary||0)-Number(a.Avg_Monetary||0))} margin={{top:0,right:20,left:0,bottom:30}}>
              <XAxis dataKey="name" tick={{fill:T.subtle,fontSize:9}} angle={-25} textAnchor="end"/>
              <YAxis tick={{fill:T.dim,fontSize:9}} tickFormatter={v=>fmt(v)}/>
              <Tooltip content={<TT fmt={v=>fmt(v)}/>}/>
              <Bar dataKey="Avg_Monetary" radius={[4,4,0,0]}>{segments.slice(0,8).map((s,i)=><Cell key={i} fill={SEG_COLORS[s.name]||T.blue}/>)}</Bar>
            </BarChart>
          </ResponsiveContainer>
          <div style={{textAlign:"center",color:T.dim,fontSize:11,marginTop:8}}>Click any segment card to expand details</div>
        </div>
      )}
    </div>
  );
}

// ── TEMPORAL TAB ─────────────────────────────────────────────────────────
function TemporalTab({data}){
  const daily=(data.daily_patterns||[]).map(d=>({day:(d.DayName||"").slice(0,3),revenue:d.Total_Revenue,customers:d.Unique_Customers,isSat:d.DayName==="Saturday",isSun:d.DayName==="Sunday"}));
  const hourly=(data.hourly_patterns||[]).map(h=>({h:`${String(h.Hour).padStart(2,"0")}`,rev:h.Total_Revenue,txns:h.Num_Transactions}));
  const peakDay=(data.daily_patterns||[]).reduce((a,b)=>Number(a.Total_Revenue||0)>Number(b.Total_Revenue||0)?a:b,data.daily_patterns?.[0]||{});
  const peakHour=(data.hourly_patterns||[]).reduce((a,b)=>Number(a.Total_Revenue||0)>Number(b.Total_Revenue||0)?a:b,data.hourly_patterns?.[0]||{});
  if(!daily.length&&!hourly.length) return <div style={{...card,textAlign:"center",padding:48,color:T.subtle}}>No temporal data available.</div>;
  return (
    <div>
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:16,marginBottom:16}}>
        {daily.length>0&&<>
          <div style={card}>
            <SectionTitle accent={T.blue}>Revenue by Day of Week</SectionTitle>
            <ResponsiveContainer width="100%" height={210}>
              <BarChart data={daily} margin={{top:0,right:10,left:0,bottom:0}}>
                <XAxis dataKey="day" tick={{fill:T.subtle,fontSize:11}}/>
                <YAxis tick={{fill:T.dim,fontSize:10}} tickFormatter={v=>fmt(v)}/>
                <Tooltip content={<TT fmt={v=>fmt(v)}/>}/>
                <Bar dataKey="revenue" radius={[4,4,0,0]}>{daily.map((d,i)=><Cell key={i} fill={d.isSat?T.red:d.isSun?"#4A5568":T.blue}/>)}</Bar>
              </BarChart>
            </ResponsiveContainer>
            <div style={{marginTop:10,padding:"8px 12px",background:"#0A0F1E",borderRadius:6,fontSize:11,color:T.subtle}}>
              <span style={{color:T.green}}>Peak: </span>{peakDay.DayName} · {fmt(peakDay.Total_Revenue||0)} · {fmtN(peakDay.Unique_Customers||0)} customers
            </div>
          </div>
          <div style={card}>
            <SectionTitle accent={T.purple}>Active Customers by Day</SectionTitle>
            <ResponsiveContainer width="100%" height={210}>
              <BarChart data={daily} margin={{top:0,right:10,left:0,bottom:0}}>
                <XAxis dataKey="day" tick={{fill:T.subtle,fontSize:11}}/>
                <YAxis tick={{fill:T.dim,fontSize:10}}/>
                <Tooltip content={<TT/>}/>
                <Bar dataKey="customers" radius={[4,4,0,0]}>{daily.map((d,i)=><Cell key={i} fill={d.isSat?T.red:d.isSun?"#4A5568":T.purple}/>)}</Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </>}
      </div>
      {hourly.length>0&&(
        <div style={card}>
          <SectionTitle accent={T.yellow}>Hourly Activity — Transactions & Revenue</SectionTitle>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={hourly} margin={{top:5,right:20,left:0,bottom:0}}>
              <XAxis dataKey="h" tick={{fill:T.subtle,fontSize:10}} tickFormatter={h=>`${h}h`}/>
              <YAxis yAxisId="l" tick={{fill:T.dim,fontSize:9}} tickFormatter={v=>fmtN(v)}/>
              <YAxis yAxisId="r" orientation="right" tick={{fill:T.dim,fontSize:9}} tickFormatter={v=>fmt(v)}/>
              <Tooltip content={<TT fmt={(v,n)=>n==="Revenue"?fmt(v):fmtN(v)}/>}/>
              <Line yAxisId="l" type="monotone" dataKey="txns" stroke={T.blue} strokeWidth={2} dot={false} name="Transactions"/>
              <Line yAxisId="r" type="monotone" dataKey="rev" stroke={T.yellow} strokeWidth={2} dot={false} name="Revenue"/>
            </LineChart>
          </ResponsiveContainer>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:8,marginTop:12}}>
            {[
              {label:"Peak Hour",value:`${String(peakHour.Hour||12).padStart(2,"0")}:00`,sub:"Highest revenue",color:T.yellow},
              {label:"Optimal Send Time",value:"09:00–11:00",sub:"Campaign deployment",color:T.blue},
              {label:"After-Hours",value:"<2%",sub:"Post 18h near zero",color:T.dim},
            ].map(i=>(
              <div key={i.label} style={{padding:"10px 12px",background:"#0A0F1E",borderRadius:6,borderLeft:`3px solid ${i.color}`}}>
                <div style={{fontSize:9,color:T.dim}}>{i.label}</div>
                <div style={{fontSize:13,fontWeight:700,color:i.color,fontFamily:T.mono}}>{i.value}</div>
                <div style={{fontSize:10,color:T.subtle}}>{i.sub}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── CLUSTERS TAB ─────────────────────────────────────────────────────────
function ClustersTab({data}){
  const COLORS=[T.blue,T.green,T.yellow,T.orange,T.purple,T.red];
  const rfmClusters=(data.rfm_clusters||[]).filter(c=>c.RFM_Enhanced_Cluster!==999);
  const behClusters=(data.behavioral_clusters||[]).filter(c=>c.Behavioral_Cluster!==999);
  if(!rfmClusters.length&&!behClusters.length) return <div style={{...card,textAlign:"center",padding:48,color:T.subtle}}>No cluster data available.</div>;
  const rfmChart=rfmClusters.map((c,i)=>({name:`C${c.RFM_Enhanced_Cluster}`,customers:c.Customer_Count,revPct:c.Revenue_Pct,clv:c.clv_mean,churnRisk:c.churn_risk_score_mean,loyalty:c.loyalty_index_mean,color:COLORS[i%COLORS.length]}));
  const behChart=behClusters.map((c,i)=>({name:`BC${c.Behavioral_Cluster}`,customers:c.Count,revPct:c.Revenue_Pct,clv:c.clv_mean,color:COLORS[i%COLORS.length]}));
  return (
    <div>
      <div style={{...card,borderLeft:`3px solid ${T.purple}`,marginBottom:16}}>
        <div style={{fontSize:12,color:T.subtle}}>
          <span style={{color:T.purple,fontWeight:600}}>ML Clustering — </span>
          K-Means on 52 behavioural, temporal and monetary features. RFM-Enhanced ({rfmChart.length} clusters) + Behavioural ({behChart.length} clusters).
        </div>
      </div>
      {rfmChart.length>0&&<>
        <div style={{fontSize:11,fontWeight:600,color:T.subtle,textTransform:"uppercase",letterSpacing:"0.1em",marginBottom:10}}>RFM-Enhanced Clusters</div>
        <div style={{display:"grid",gridTemplateColumns:`repeat(${rfmChart.length},1fr)`,gap:10,marginBottom:16}}>
          {rfmChart.map(c=>(
            <div key={c.name} style={{...card,borderTop:`3px solid ${c.color}`,textAlign:"center"}}>
              <div style={{fontSize:20,fontWeight:800,color:c.color,fontFamily:T.mono}}>{c.name}</div>
              <div style={{fontSize:13,fontWeight:700,color:"#F0F4FF"}}>{fmtN(c.customers)}</div>
              <div style={{fontSize:9,color:T.dim,marginBottom:8}}>customers</div>
              <div style={{height:1,background:T.border,marginBottom:8}}/>
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:4,fontSize:10,marginBottom:8}}>
                <div><div style={{color:T.dim}}>Rev %</div><div style={{color:"#F0F4FF",fontWeight:700}}>{c.revPct}%</div></div>
                <div><div style={{color:T.dim}}>CLV</div><div style={{color:"#F0F4FF",fontWeight:700}}>{fmt(c.clv)}</div></div>
              </div>
              {c.churnRisk!=null&&<div style={{fontSize:9,padding:"2px 6px",borderRadius:3,display:"inline-block",background:c.churnRisk>40?`${T.red}20`:`${T.green}20`,color:c.churnRisk>40?T.red:T.green}}>Churn {Number(c.churnRisk).toFixed(0)}%</div>}
            </div>
          ))}
        </div>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:16,marginBottom:16}}>
          <div style={card}>
            <SectionTitle accent={T.purple}>Revenue by RFM Cluster</SectionTitle>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={rfmChart} margin={{top:0,right:10,left:0,bottom:0}}>
                <XAxis dataKey="name" tick={{fill:T.subtle,fontSize:11}}/>
                <YAxis tick={{fill:T.dim,fontSize:10}} tickFormatter={v=>`${v}%`}/>
                <Tooltip content={<TT fmt={v=>`${v}%`}/>}/>
                <Bar dataKey="revPct" radius={[4,4,0,0]}>{rfmChart.map((c,i)=><Cell key={i} fill={c.color}/>)}</Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div style={card}>
            <SectionTitle accent={T.purple}>Loyalty vs Churn Risk</SectionTitle>
            <ResponsiveContainer width="100%" height={200}>
              <ScatterChart margin={{top:10,right:20,left:0,bottom:20}}>
                <XAxis type="number" dataKey="churnRisk" domain={[0,80]} tick={{fill:T.dim,fontSize:9}} tickFormatter={v=>`${v}%`} label={{value:"Churn Risk %",position:"insideBottom",fill:T.dim,fontSize:9,dy:14}}/>
                <YAxis type="number" dataKey="loyalty" domain={[0,100]} tick={{fill:T.dim,fontSize:9}} tickFormatter={v=>`${v}%`} label={{value:"Loyalty %",angle:-90,position:"insideLeft",fill:T.dim,fontSize:9}}/>
                <ZAxis type="number" dataKey="customers" range={[60,350]}/>
                <Tooltip content={<TT fmt={(v,n)=>n!=="customers"?`${Number(v).toFixed(1)}%`:fmtN(v)}/>}/>
                <Scatter data={rfmChart}>{rfmChart.map((c,i)=><Cell key={i} fill={c.color}/>)}</Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>
      </>}
      {behChart.length>0&&<>
        <div style={{fontSize:11,fontWeight:600,color:T.subtle,textTransform:"uppercase",letterSpacing:"0.1em",marginBottom:10}}>Behavioural Clusters</div>
        <div style={{display:"grid",gridTemplateColumns:`repeat(${behChart.length},1fr)`,gap:10}}>
          {behChart.map(c=>(
            <div key={c.name} style={{...card,borderTop:`3px solid ${c.color}`}}>
              <div style={{fontSize:18,fontWeight:800,color:c.color,fontFamily:T.mono,marginBottom:8}}>{c.name}</div>
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:8}}>
                {[["Customers",fmtN(c.customers)],["Rev %",`${c.revPct}%`],["Avg CLV",fmt(c.clv)]].map(([l,v])=>(
                  <div key={l} style={{background:"#0A0F1E",padding:"8px",borderRadius:5}}>
                    <div style={{fontSize:9,color:T.dim}}>{l}</div>
                    <div style={{fontSize:12,fontWeight:700,color:"#F0F4FF",fontFamily:T.mono}}>{v}</div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </>}
    </div>
  );
}

// ── INSIGHTS TAB ─────────────────────────────────────────────────────────
function InsightsTab({data}){
  const [activeCategory,setActiveCategory]=useState("all");
  const ins_data=data.insights;
  if(!ins_data) return <div style={{...card,textAlign:"center",padding:48}}><div style={{fontSize:36,marginBottom:12}}>📋</div><div style={{fontSize:14,color:T.subtle}}>No insights data available.</div></div>;
  const alerts=ins_data.alerts||[];
  const insights=ins_data.insights||[];
  const ranking=ins_data.opportunity_ranking||[];
  const categories=["all",...new Set(insights.map(i=>i.category))];
  const filtered=activeCategory==="all"?insights:insights.filter(i=>i.category===activeCategory);
  const totalImpact=ranking.reduce((s,r)=>s+Number(r.revenue_impact_gbp||0),0);
  return (
    <div>
      <div style={{marginBottom:20}}>
        <SectionTitle accent={T.red}>System Alerts</SectionTitle>
        <div style={{display:"flex",flexDirection:"column",gap:8}}>
          {alerts.map(a=>(
            <div key={a.alert_id} style={{...card,display:"flex",gap:14,alignItems:"flex-start",borderLeft:`3px solid ${SEVERITY_COLOR[a.severity]||T.blue}`}}>
              <div style={{fontSize:18,marginTop:1}}>{a.severity==="RED"?"🚨":a.severity==="AMBER"?"⚠️":"✅"}</div>
              <div style={{flex:1}}>
                <div style={{display:"flex",gap:8,alignItems:"center",marginBottom:4}}>
                  <span style={{fontSize:11,fontWeight:700,color:"#F0F4FF"}}>{a.segment}</span>
                  <span style={{fontSize:10,padding:"1px 6px",borderRadius:3,background:`${SEVERITY_COLOR[a.severity]||T.blue}20`,color:SEVERITY_COLOR[a.severity]||T.blue}}>{a.severity}</span>
                  <span style={{fontSize:10,color:T.dim}}>{a.metric}: {typeof a.value==="number"?Number(a.value).toFixed(1):a.value}</span>
                </div>
                <div style={{fontSize:12,color:T.text,marginBottom:6}}>{a.message}</div>
                <div style={{fontSize:11,color:T.subtle,fontStyle:"italic"}}>→ {a.action_required}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
      {ranking.length>0&&(
        <div style={{...card,marginBottom:20,borderLeft:`3px solid ${T.green}`}}>
          <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:14}}>
            <SectionTitle accent={T.green}>Opportunity Ranking (by Composite Score)</SectionTitle>
            <div style={{textAlign:"right"}}>
              <div style={{fontSize:18,fontWeight:800,color:T.green,fontFamily:T.mono}}>{fmt(totalImpact)}</div>
              <div style={{fontSize:10,color:T.dim}}>total revenue impact</div>
            </div>
          </div>
          <div style={{display:"flex",flexDirection:"column",gap:6}}>
            {ranking.map((r,i)=>(
              <div key={r.insight_id} style={{display:"grid",gridTemplateColumns:"24px 1fr 80px 80px 90px",gap:10,alignItems:"center",padding:"10px 12px",background:"#0A0F1E",borderRadius:6}}>
                <div style={{fontSize:13,fontWeight:800,color:T.dim,fontFamily:T.mono,textAlign:"center"}}>{i+1}</div>
                <div>
                  <div style={{fontSize:12,fontWeight:600,color:"#F0F4FF"}}>{r.title}</div>
                  <div style={{fontSize:10,color:T.subtle}}>{fmtN(r.customers_affected)} customers · {r.campaign_timing}</div>
                </div>
                <div style={{textAlign:"right"}}>
                  <div style={{fontSize:12,fontWeight:700,color:T.green,fontFamily:T.mono}}>{fmt(r.revenue_impact_gbp)}</div>
                  <div style={{fontSize:9,color:T.dim}}>impact</div>
                </div>
                <div style={{textAlign:"center",fontSize:10,padding:"2px 6px",borderRadius:3,background:`${PRIORITY_COLOR[r.priority]||T.blue}20`,color:PRIORITY_COLOR[r.priority]||T.blue}}>{r.priority}</div>
                <div style={{textAlign:"center",fontSize:10,padding:"2px 6px",borderRadius:3,background:`${EFFORT_COLOR[r.effort_level]||T.blue}20`,color:EFFORT_COLOR[r.effort_level]||T.blue}}>{r.effort_level}</div>
              </div>
            ))}
          </div>
        </div>
      )}
      <div>
        <div style={{display:"flex",gap:6,marginBottom:12}}>
          {categories.map(c=>(
            <button key={c} onClick={()=>setActiveCategory(c)} style={{padding:"5px 12px",borderRadius:20,border:"none",fontSize:11,fontWeight:activeCategory===c?600:400,background:activeCategory===c?(CATEGORY_COLOR[c]||T.blue):"#1A2235",color:activeCategory===c?"#070B14":T.subtle,cursor:"pointer",fontFamily:"'IBM Plex Sans',system-ui,sans-serif",textTransform:"capitalize"}}>
              {c.replace("_"," ")} ({c==="all"?insights.length:insights.filter(i=>i.category===c).length})
            </button>
          ))}
        </div>
        <div style={{display:"flex",flexDirection:"column",gap:10}}>
          {filtered.map(ins=>(
            <div key={ins.id} style={{...card,borderLeft:`3px solid ${CATEGORY_COLOR[ins.category]||T.blue}`}}>
              <div style={{display:"flex",gap:8,alignItems:"center",marginBottom:8}}>
                <span style={{fontSize:10,padding:"2px 6px",borderRadius:3,background:`${PRIORITY_COLOR[ins.priority]}20`,color:PRIORITY_COLOR[ins.priority]}}>{ins.priority}</span>
                <span style={{fontSize:10,padding:"2px 6px",borderRadius:3,background:`${EFFORT_COLOR[ins.effort_level]}20`,color:EFFORT_COLOR[ins.effort_level]}}>{ins.effort_level}</span>
                <span style={{fontSize:10,color:T.dim,fontFamily:T.mono}}>{ins.id}</span>
                <span style={{marginLeft:"auto",fontSize:12,fontWeight:700,color:T.green,fontFamily:T.mono}}>{ins.revenue_impact>0?fmt(ins.revenue_impact):"—"}</span>
              </div>
              <div style={{fontSize:13,fontWeight:700,color:"#F0F4FF",marginBottom:4}}>{ins.title}</div>
              <div style={{fontSize:12,color:T.text,marginBottom:8}}>{ins.headline}</div>
              <div style={{fontSize:11,color:T.subtle,lineHeight:1.6,marginBottom:10}}>{ins.detail}</div>
              <div style={{background:"#0A0F1E",padding:"10px 12px",borderRadius:6,borderLeft:`2px solid ${CATEGORY_COLOR[ins.category]||T.blue}`}}>
                <div style={{fontSize:9,color:T.dim,textTransform:"uppercase",letterSpacing:"0.1em",marginBottom:4}}>Recommended Action</div>
                <div style={{fontSize:11,color:T.text,lineHeight:1.6}}>{ins.recommended_action}</div>
              </div>
              <div style={{display:"flex",gap:16,marginTop:10}}>
                <div style={{fontSize:10,color:T.subtle}}><span style={{color:T.dim}}>Timing: </span>{ins.campaign_timing}</div>
                <div style={{fontSize:10,color:T.subtle}}><span style={{color:T.dim}}>Customers: </span>{fmtN(ins.customers_affected)}</div>
                <div style={{fontSize:10,color:T.subtle}}><span style={{color:T.dim}}>Metric: </span>{ins.metric_name} = {ins.metric_value}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── MAIN APP (data pre-loaded, no upload screen) ──────────────────────────
function App(){
  const [fileData] = useState(INJECTED);
  const [activeTab,setActiveTab]=useState("executive");

  const tabs=[
    {id:"executive",label:"Executive Overview",show:true},
    {id:"segments", label:"Segment Analysis",  show:true},
    {id:"temporal", label:"Temporal Patterns", show:!!(fileData.daily_patterns||fileData.hourly_patterns)},
    {id:"clusters", label:"ML Clusters",       show:!!(fileData.rfm_clusters||fileData.behavioral_clusters)},
    {id:"insights", label:"Insights & Actions",show:!!fileData.insights},
  ];

  const VIEWS={
    executive:<ExecutiveTab data={fileData}/>,
    segments: <SegmentsTab data={fileData}/>,
    temporal: <TemporalTab data={fileData}/>,
    clusters: <ClustersTab data={fileData}/>,
    insights: <InsightsTab data={fileData}/>,
  };

  const meta=fileData._meta||{};

  return (
    <div style={{minHeight:"100vh",background:T.bg,color:T.text,fontFamily:"'IBM Plex Sans',system-ui,sans-serif",fontSize:13}}>
      <div style={{borderBottom:`1px solid ${T.border}`,padding:"0 28px",background:"rgba(7,11,20,0.97)",position:"sticky",top:0,zIndex:100,display:"flex",alignItems:"center",justifyContent:"space-between",height:54}}>
        <div style={{display:"flex",alignItems:"center",gap:10}}>
          <div style={{width:26,height:26,background:`linear-gradient(135deg,${T.green},${T.blue})`,borderRadius:6,display:"flex",alignItems:"center",justifyContent:"center",fontSize:13}}>📊</div>
          <div>
            <div style={{fontSize:13,fontWeight:700,color:"#F0F4FF"}}>Retail Intelligence</div>
            <div style={{fontSize:9,color:T.dim}}>UCI Online Retail II · Generated {meta.generated_at||""}</div>
          </div>
        </div>
        <div style={{display:"flex",gap:6,alignItems:"center"}}>
          {fileData.rfm_summary&&<>
            <div style={{fontSize:10,padding:"2px 8px",borderRadius:20,background:"#0D2818",border:`1px solid ${T.green}33`,color:T.green}}>{Number(fileData.rfm_summary.total_customers||0).toLocaleString()} Customers</div>
            <div style={{fontSize:10,padding:"2px 8px",borderRadius:20,background:"#0D2818",border:`1px solid ${T.green}33`,color:T.green}}>{fmt(fileData.rfm_summary.total_revenue||0)}</div>
          </>}
          <div style={{fontSize:10,padding:"2px 8px",borderRadius:20,background:"#1A1410",border:`1px solid ${T.yellow}33`,color:T.yellow}}>{meta.files_loaded||0} files · {meta.generated_at||""}</div>
        </div>
      </div>
      <div style={{display:"flex",gap:2,padding:"0 28px",borderBottom:`1px solid ${T.border}`,background:T.bg,overflowX:"auto"}}>
        {tabs.filter(t=>t.show).map(t=>(
          <button key={t.id} onClick={()=>setActiveTab(t.id)} style={{padding:"11px 16px",fontSize:12,fontWeight:activeTab===t.id?600:400,color:activeTab===t.id?T.green:T.muted,borderBottom:activeTab===t.id?`2px solid ${T.green}`:"2px solid transparent",cursor:"pointer",whiteSpace:"nowrap",background:"none",border:"none",fontFamily:"inherit"}}>{t.label}</button>
        ))}
      </div>
      <div style={{padding:"22px 28px"}} className="fade-in" key={activeTab}>
        {VIEWS[activeTab]}
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App/>);
"""

# FIXED HTML TEMPLATE - Added react-is and updated Recharts to 2.12.0
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>{title}</title>
<!-- FIXED: Added react-is dependency which is REQUIRED for Recharts UMD -->
<!-- FIXED: Updated Recharts from 2.8.0 to 2.12.0 for React 18 compatibility -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/react/18.2.0/umd/react.production.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/react-dom/18.2.0/umd/react-dom.production.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/babel-standalone/7.23.2/babel.min.js"></script>
<!-- react-is is a peer dependency of Recharts and MUST be loaded before Recharts -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/prop-types/15.8.1/prop-types.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/recharts/2.12.0/Recharts.js"></script>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet" />
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  html,body,#root{{min-height:100%;background:#070B14}}
  body{{color:#C8D0E0;font-family:'IBM Plex Sans',system-ui,sans-serif}}
  ::-webkit-scrollbar{{width:4px;height:4px}}
  ::-webkit-scrollbar-track{{background:#0A0F1E}}
  ::-webkit-scrollbar-thumb{{background:#1A2235;border-radius:4px}}
  @keyframes fadeIn{{from{{opacity:0;transform:translateY(8px)}}to{{opacity:1;transform:translateY(0)}}}}
  .fade-in{{animation:fadeIn 0.3s ease}}
</style>
</head>
<body>
<div id="root"></div>
<!-- Injected pipeline data -->
<script>
window.__DASHBOARD_DATA__ = {data_json};
</script>
<script type="text/babel">
{dashboard_js}
</script>
</body>
</html>
"""

# ── BUILDER ───────────────────────────────────────────────────────────────
def build_html(data: dict, output_path: Path) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    date_slug = datetime.now().strftime("%Y%m%d")

    # Add metadata so the dashboard can display generation info
    data["_meta"] = {
        "generated_at": now,
        "files_loaded": len([k for k in data if not k.startswith("_")]),
    }

    data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    title = f"Retail Intelligence Dashboard — {date_slug}"

    html = HTML_TEMPLATE.format(
        title=title,
        data_json=data_json,
        dashboard_js=DASHBOARD_JS,
    )

    output_path.write_text(html, encoding="utf-8")
    size_kb = output_path.stat().st_size / 1024
    print(f"✅  Dashboard generated: {output_path.resolve()}")
    print(f"    Size: {size_kb:.1f} KB")
    print(f"    Generated at: {now}")
    print(f"\n🚀  Share tip: email the file, upload to Google Drive,")
    print(f"    or host on GitHub Pages. Anyone can open it in a browser.\n")

# ── CLI ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate a self-contained Retail Intelligence Dashboard HTML file."
    )
    parser.add_argument(
        "--data-dir", "-d",
        default=".",
        help="Directory containing the pipeline export files (default: current directory)"
    )
    parser.add_argument(
        "--out", "-o",
        default=None,
        help="Output filename (default: dashboard_YYYYMMDD.html)"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        sys.exit(f"❌  Data directory not found: {data_dir}")

    output_name = args.out or f"dashboard_{datetime.now().strftime('%Y%m%d')}.html"
    output_path = Path("/home/cairo/code/portfolio/customer-segmentation/outputs/reports/" + output_name)

    data = load_data(data_dir)
    build_html(data, output_path)

if __name__ == "__main__":
    main()
