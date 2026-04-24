const BACKEND='http://localhost:8000';
const drop=document.getElementById('drop'), fileInput=document.getElementById('file');
const upload=document.getElementById('upload'), record=document.getElementById('record');
const label=document.getElementById('label'), result=document.getElementById('result');
const clsEl=document.getElementById('cls'), confEl=document.getElementById('conf'), solEl=document.getElementById('sol');
drop.addEventListener('click', ()=>fileInput.click()); fileInput.addEventListener('change', ()=>handle(fileInput.files[0])); let current=null;
function handle(f){ current=f; drop.innerText=f.name; }
upload.addEventListener('click', async ()=>{ if(!current){ alert('select file'); return;} const fd=new FormData(); fd.append('file', current); if(label.value) fd.append('label', label.value); upload.disabled=true; try{ const res=await fetch(BACKEND+'/predict',{method:'POST', body:fd}); const data=await res.json(); const r = data.result || data; clsEl.innerText='Class: '+r.class; confEl.innerText='Confidence: '+(r.confidence*100).toFixed(1)+'%'; solEl.innerText = r.solution ? r.solution.temporary : ''; result.hidden=false;}catch(e){alert(e.message)} finally{upload.disabled=false}});
record.addEventListener('click', async ()=>{ if(window.rec && window.rec.state==='recording'){ window.rec.stop(); record.innerText='Record'; return;} const s=await navigator.mediaDevices.getUserMedia({audio:true}); const rec=new MediaRecorder(s); window.rec=rec; let chunks=[]; rec.ondataavailable=e=>chunks.push(e.data); rec.onstop=()=>{ const blob=new Blob(chunks,{type:'audio/webm'}); const file=new File([blob],'recording.webm',{type:'audio/webm'}); handle(file); }; rec.start(); record.innerText='Stop'; });
