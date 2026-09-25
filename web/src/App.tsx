import React, { useState, useEffect, useRef } from 'react';
import { Upload, Play, Save } from 'lucide-react';
import { createClient } from '@supabase/supabase-js';

// Initialize Supabase client
const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || '';
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY || '';
const supabase = supabaseUrl && supabaseAnonKey ? createClient(supabaseUrl, supabaseAnonKey) : null;

function App() {
  const [imageUrl, setImageUrl] = useState<string>('');
  const [refParam, setRefParam] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [result, setResult] = useState<any>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    // Read ?image= and ?ref= from URL
    const params = new URLSearchParams(window.location.search);
    const imgParam = params.get('image');
    if (imgParam) {
      setImageUrl(imgParam);
    }
    const ref = params.get('ref');
    if (ref) {
      setRefParam(ref);
    }
  }, []);

  const handleSegment = async () => {
    if (!imageUrl) return;
    setLoading(true);
    setResult(null);
    try {
      const res = await fetch(imageUrl);
      const blob = await res.blob();
      
      const formData = new FormData();
      formData.append('file', blob, 'image.png');
      formData.append('params', JSON.stringify({
        model_name: 'cpsam_v2',
        gpu: false, 
      }));

      // In production, point this to your actual deployed backend URL (e.g., via another env var)
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const apiRes = await fetch(`${apiUrl}/segment`, {
        method: 'POST',
        body: formData
      });
      
      if (!apiRes.ok) throw new Error(await apiRes.text());
      const data = await apiRes.json();
      setResult(data);
    } catch (e) {
      console.error(e);
      alert('Error during segmentation: ' + e);
    } finally {
      setLoading(false);
    }
  };

  const handleSaveToDatabase = async () => {
    if (!supabase) {
      alert('Supabase credentials not configured in environment variables.');
      return;
    }
    if (!result) return;
    
    setSaving(true);
    try {
      // Assuming a 'segmentations' table exists in AstroBotany database
      const { data, error } = await supabase
        .from('segmentations')
        .insert([
          { 
            image_url: imageUrl, 
            ref: refParam, 
            polygon_count: result.polygons.length,
            polygons: result.polygons // storing the raw JSON
          }
        ]);
        
      if (error) throw error;
      alert('Saved successfully to database!');
    } catch (e: any) {
      console.error(e);
      alert('Error saving to database: ' + e.message);
    } finally {
      setSaving(false);
    }
  };

  useEffect(() => {
    if (result && canvasRef.current && imageRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      if (!ctx) return;
      
      const img = imageRef.current;
      canvasRef.current.width = img.naturalWidth;
      canvasRef.current.height = img.naturalHeight;
      
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      ctx.drawImage(img, 0, 0);
      
      // Draw polygons
      ctx.lineWidth = 2;
      result.polygons.forEach((poly: any) => {
        ctx.strokeStyle = `hsl(${(poly.id * 137.5) % 360}, 100%, 50%)`;
        ctx.beginPath();
        poly.points.forEach((p: any, i: number) => {
          if (i === 0) ctx.moveTo(p.x, p.y);
          else ctx.lineTo(p.x, p.y);
        });
        ctx.closePath();
        ctx.stroke();
      });
    }
  }, [result]);

  return (
    <div style={{ padding: 20, fontFamily: 'sans-serif' }}>
      <header style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 20 }}>
        <h1>CoSE Cell Segmenter (Web)</h1>
      </header>
      
      <div style={{ display: 'flex', gap: 10, marginBottom: 20 }}>
        <input 
          type="text" 
          placeholder="Image URL..." 
          value={imageUrl} 
          onChange={e => setImageUrl(e.target.value)}
          style={{ width: 400, padding: 8 }}
        />
        <button onClick={handleSegment} disabled={loading || !imageUrl} style={{ display: 'flex', alignItems: 'center', gap: 5, padding: 8 }}>
          <Play size={16} />
          {loading ? 'Segmenting...' : 'Run Cellpose'}
        </button>
      </div>

      <div style={{ position: 'relative', border: '1px solid #ccc', minHeight: 400, display: 'inline-block' }}>
        {imageUrl && (
          <img 
            ref={imageRef}
            src={imageUrl} 
            alt="Source" 
            crossOrigin="anonymous"
            style={{ display: result ? 'none' : 'block', maxWidth: '100%', maxHeight: '80vh' }}
          />
        )}
        <canvas 
          ref={canvasRef} 
          style={{ display: result ? 'block' : 'none', maxWidth: '100%', maxHeight: '80vh' }} 
        />
      </div>
      
      {result && (
        <div style={{ marginTop: 20 }}>
          <p>Found {result.polygons.length} objects.</p>
          <button onClick={handleSaveToDatabase} disabled={saving} style={{ display: 'flex', alignItems: 'center', gap: 5, padding: 8 }}>
            <Save size={16} /> {saving ? 'Saving...' : 'Save to Database'}
          </button>
        </div>
      )}
    </div>
  );
}

export default App;
