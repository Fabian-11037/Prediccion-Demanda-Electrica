// Cálculos sobre resultados conservados. Las semillas se agregan después de evaluar.
export const mean=a=>a.reduce((s,x)=>s+x,0)/a.length;
export function windowMetrics(pred,start,count,q){
 if(!Number.isInteger(start)||start<0||!Number.isInteger(count)||count<1||start>=pred.mae_daily.length)throw new Error('Origen fuera del periodo disponible.');
 const end=Math.min(start+count,pred.mae_daily.length),n=end-start;
 const mae=mean(pred.mae_daily.slice(start,end));
 const rmse=mean(pred.mse_daily.map(seed=>Math.sqrt(mean(seed.slice(start,end)))));
 return {mae,rmse,mase:mae/q,n,end};
}
export function rank(a){
 const sorted=a.map((v,i)=>({v,i})).sort((a,b)=>a.v-b.v),out=Array(a.length);
 for(let i=0;i<sorted.length;){let j=i+1;while(j<sorted.length&&sorted[j].v===sorted[i].v)j++;const v=(i+j-1)/2+1;for(let k=i;k<j;k++)out[sorted[k].i]=v;i=j;}
 return out;
}
export function spearman(x,y){
 const pairs=x.map((v,i)=>[v,y[i]]).filter(a=>a.every(Number.isFinite));if(pairs.length<3)return null;
 const a=rank(pairs.map(a=>a[0])),b=rank(pairs.map(a=>a[1])),ma=mean(a),mb=mean(b);
 let ab=0,aa=0,bb=0;for(let i=0;i<a.length;i++){ab+=(a[i]-ma)*(b[i]-mb);aa+=(a[i]-ma)**2;bb+=(b[i]-mb)**2;}
 return aa&&bb?ab/Math.sqrt(aa*bb):null;
}
export function calendarProfile(calendar,start,end){
 return Array.from({length:7},(_,d)=>Array.from({length:24},(_,h)=>{
  let sum=0,n=0;for(let y=start;y<=end;y++){const row=calendar[y]?.[d];if(row){sum+=row.mean[h]*row.count[h];n+=row.count[h];}}
  return n?sum/n:null;
 }));
}
export function pairRecord(rows,a,b){return rows.find(r=>(r.A===a&&r.B===b)||(r.A===b&&r.B===a))??null;}
export function temporalVerdict(row){
 if(!row)return 'Selecciona dos procedimientos distintos.';
 if(row.p_TOST_Holm<.05)return 'Equivalencia dentro del margen relativo del 3 %.';
 return row.p_DM_Holm<.05?'Se detecta una diferencia temporal.':'No se detecta una diferencia temporal con este contraste.';
}
export function escapeHTML(value){return String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
