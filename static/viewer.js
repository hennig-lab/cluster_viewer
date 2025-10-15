document.addEventListener('DOMContentLoaded', () => {

  async function loadData() {
    const loading = document.getElementById('loading');
    const grid = document.getElementById('grid');

    // Show loading message
    loading.style.display = 'block';
    grid.style.display = 'none';

    const res = await fetch('/api/neurons');
    const neurons = await res.json();
    grid.innerHTML = '';

    // Create cards one by one
    for (const [idx, n] of neurons.entries()) {
        const card = document.createElement('div');
        card.className = 'card' + (n.excluded ? ' excluded' : '');
        card.title = `${n.filename} | cluster ${n.cluster_id}`;
        card.dataset.filename = n.filename;
        card.dataset.clusterId = n.cluster_id;
        card.onclick = async () => {
            const res = await fetch('/api/toggle', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ filename: n.filename, cluster_id: n.cluster_id })
            });
            const data = await res.json();
            const excludedSet = new Set(data.excluded.map(e => `${e[0]}_${e[1]}`));

            // Loop through all cards and update excluded class
            document.querySelectorAll('.card').forEach(cardEl => {
                const fname = cardEl.dataset.filename;
                const cid = parseInt(cardEl.dataset.clusterId);
                const key = `${fname}_${cid}`;
                if (excludedSet.has(key)) {
                    cardEl.classList.add('excluded');
                } else {
                    cardEl.classList.remove('excluded');
                }
            });
        };

        // Add a visible title
        const titleEl = document.createElement('div');
        titleEl.innerText = `${n.filename} - cluster ${n.cluster_id} (${n.firing_rate_hz.toFixed(1)} Hz)`;
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
            type: 'bar',
            data: {
            labels: n.ISI_bins.map(v => v.toFixed(2)),
            datasets: [{ data: n.ISI_freqs, borderColor: '#36a2eb', fill: false, backgroundColor: 'black', pointRadius: 0 }]
            },
            options: {
                responsive: true,
                // maintainAspectRatio: false,
                scales: {
                    x: { type: 'logarithmic', title: { display: true, text: 'ISI (ms)' } },
                    y: {
                        display: true,
                        title: { display: true, text: 'Proportion' },
                        ticks: {
                            maxTicksLimit: 3,
                            callback: function(value, index, ticks) {
                                return value.toFixed(2); // round to 2 decimal places
                            }
                        }
                    }
                },
                plugins: { legend: { display: false } }
            }
        });

        const labels = [...Array(64).keys()];
        const numQuintiles = n.waveform_quintiles.length; // should be 10
        const medianIdx = Math.floor(numQuintiles / 2);   // index 5 (50th percentile)

        const datasets = n.waveform_quintiles.map((quintile, idx) => {
        let shade;
        if (idx === medianIdx) {
            shade = 0; // black
        } else {
            // distance from median determines lightness (0 = black, 200 = light gray)
            const dist = Math.abs(idx - medianIdx);
            const maxDist = medianIdx;
            shade = Math.round(50 + (dist / maxDist) * 150); // 50 → 200
        }
        const color = `rgb(${shade}, ${shade}, ${shade})`;
        return { data: quintile, borderColor: color, fill: false, pointRadius: 0 };
        });

        new Chart(c2, {
            type: 'line',
            data: {
                labels: labels,
                datasets: datasets
            },
            options: {
                responsive: true,
                // maintainAspectRatio: false,
                aspectRatio: 1.2,
                scales: {
                    x: { display: false },
                    y: {
                        display: true,
                        title: { display: true, text: 'Potential (mV)' },
                        ticks: {
                        // display roughly 5 ticks automatically
                        maxTicksLimit: 5
                        }
                    }
                    },
                plugins: { legend: { display: false } }
            }
        });

        // Let browser update DOM before next iteration
        await new Promise(r => requestAnimationFrame(r));
    }

    // Hide loading, show grid
    loading.style.display = 'none';
    grid.style.display = 'grid';
  }

  loadData();
});
