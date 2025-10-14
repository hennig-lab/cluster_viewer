document.addEventListener('DOMContentLoaded', () => {

  async function loadData() {
    const res = await fetch('/api/neurons');
    const neurons = await res.json();
    const grid = document.getElementById('grid');
    grid.innerHTML = '';

    neurons.forEach((n, idx) => {
        const card = document.createElement('div');
        card.className = 'card' + (n.excluded ? ' excluded' : '');
        card.title = `${n.filename} | cluster ${n.cluster_id}`;
        card.onclick = async () => {
        await fetch('/api/toggle', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ filename: n.filename, cluster_id: n.cluster_id })
        });
        loadData();
        };

        // Add a visible title
        const titleEl = document.createElement('div');
        titleEl.innerText = `${n.filename} - cluster ${n.cluster_id}`;
        titleEl.style.fontSize = '12px';
        titleEl.style.marginBottom = '4px';
        titleEl.style.fontWeight = 'bold';
        card.appendChild(titleEl);

        const c1 = document.createElement('canvas');
        const c2 = document.createElement('canvas');
        card.appendChild(c1);
        card.appendChild(c2);
        grid.appendChild(card);

        new Chart(c1, {
            type: 'line',
            data: {
            labels: n.ISI_bins.map(v => v.toFixed(0)),
            datasets: [{ data: n.ISI_freqs, borderColor: '#36a2eb', fill: false }]
            },
            options: {
            responsive: true,
            // maintainAspectRatio: false,
            scales: {
                x: { type: 'logarithmic', title: { display: true, text: 'ISI (ms)' } },
                y: { display: false }
            },
            plugins: { legend: { display: false } }
            }
        });

        const avg = n.waveform_quintiles.reduce((a,b)=>a.map((v,i)=>v+b[i]), new Array(64).fill(0))
            .map(v=>v/n.waveform_quintiles.length);
        new Chart(c2, {
            type: 'line',
            data: {
            labels: [...Array(64).keys()],
            datasets: [{ data: avg, borderColor: '#4caf50', fill: false }]
            },
            options: {
            responsive: true,
            // maintainAspectRatio: false,
            scales: { x: { display: false }, y: { display: false } },
            plugins: { legend: { display: false } }
            }
        });    
    });
  }

  loadData();
});
