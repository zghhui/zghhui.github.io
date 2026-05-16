(function () {
    'use strict';

    // ===== State =====
    let editor = null;
    let currentProblem = null;
    let timerInterval = null;
    let timerSeconds = 0;
    let currentView = 'practice';

    // ===== Storage =====
    const STORAGE_KEY = 'algo_practice_data';

    function loadData() {
        try {
            return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {};
        } catch { return {}; }
    }

    function saveData(data) {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
    }

    function getProblemData(id) {
        const data = loadData();
        return data[id] || { code: null, bestScore: null, attempts: 0, lastAttempt: null, reviews: [] };
    }

    function saveProblemData(id, update) {
        const data = loadData();
        data[id] = { ...getProblemData(id), ...update };
        saveData(data);
    }

    // ===== Init =====
    function init() {
        initEditor();
        renderProblemList();
        bindEvents();
        restoreTheme();
    }

    // ===== Editor =====
    function initEditor() {
        const editorEl = document.getElementById('codeEditor');
        editor = CodeMirror(editorEl, {
            mode: 'python',
            theme: 'dracula',
            lineNumbers: true,
            matchBrackets: true,
            autoCloseBrackets: true,
            styleActiveLine: true,
            indentUnit: 4,
            tabSize: 4,
            indentWithTabs: false,
            extraKeys: {
                'Tab': cm => {
                    if (cm.somethingSelected()) {
                        cm.indentSelection('add');
                    } else {
                        cm.replaceSelection('    ', 'end');
                    }
                },
                'Shift-Tab': cm => cm.indentSelection('subtract'),
                'Ctrl-/': 'toggleComment',
                'Cmd-/': 'toggleComment'
            },
            placeholder: '// 在这里写你的代码...'
        });
        editor.setValue('# 从左侧选择一道题目开始\n');
    }

    // ===== Problem List =====
    function renderProblemList(categoryFilter, difficultyFilter) {
        const list = document.getElementById('problemList');
        const cat = categoryFilter || document.getElementById('categoryFilter').value;
        const diff = difficultyFilter || document.getElementById('difficultyFilter').value;

        let problems = window.PROBLEMS;
        if (cat !== 'all') problems = problems.filter(p => p.category === cat);
        if (diff !== 'all') problems = problems.filter(p => p.difficulty === diff);

        list.innerHTML = problems.map(p => {
            const data = getProblemData(p.id);
            const isActive = currentProblem && currentProblem.id === p.id;
            const isCompleted = data.bestScore && data.bestScore >= 70;
            const diffLabel = { easy: '基础', medium: '中等', hard: '困难' }[p.difficulty];

            return `<li class="problem-item ${isActive ? 'active' : ''} ${isCompleted ? 'completed' : ''}"
                        data-id="${p.id}">
                <div class="problem-number">${isCompleted ? '✓' : p.id}</div>
                <div class="problem-info">
                    <div class="problem-name">${p.title}</div>
                    <div class="problem-meta">
                        <span><span class="difficulty-dot ${p.difficulty}"></span>${diffLabel}</span>
                        ${data.bestScore ? `<span>最高: ${data.bestScore}分</span>` : ''}
                    </div>
                </div>
            </li>`;
        }).join('');
    }

    function selectProblem(id) {
        const problem = window.PROBLEMS.find(p => p.id === id);
        if (!problem) return;

        currentProblem = problem;

        // Update title & tags
        document.getElementById('problemTitle').textContent = problem.title;
        const tagsEl = document.getElementById('problemTags');
        const diffClass = `tag-difficulty-${problem.difficulty}`;
        const diffLabel = { easy: '基础', medium: '中等', hard: '困难' }[problem.difficulty];
        tagsEl.innerHTML = `
            <span class="tag tag-category">${problem.category}</span>
            <span class="tag ${diffClass}">${diffLabel}</span>
        `;

        // Update description
        const body = document.getElementById('problemBody');
        body.innerHTML = marked.parse(problem.description);

        if (problem.keyPoints && problem.keyPoints.length > 0) {
            body.innerHTML += `
                <div class="key-points">
                    <h4>核心考点</h4>
                    <ul>${problem.keyPoints.map(p => `<li>${p}</li>`).join('')}</ul>
                </div>
            `;
        }

        // Load code
        const savedData = getProblemData(id);
        editor.setValue(savedData.code || problem.template);
        editor.clearHistory();

        // Ensure panel is expanded
        document.getElementById('problemPanel').classList.remove('collapsed');

        // Reset timer
        resetTimer();
        startTimer();

        // Update sidebar active state
        renderProblemList();

        // Clear review
        resetReview();
    }

    // ===== Timer =====
    function startTimer() {
        stopTimer();
        timerSeconds = 0;
        updateTimerDisplay();
        timerInterval = setInterval(() => {
            timerSeconds++;
            updateTimerDisplay();
        }, 1000);
    }

    function stopTimer() {
        if (timerInterval) {
            clearInterval(timerInterval);
            timerInterval = null;
        }
    }

    function resetTimer() {
        stopTimer();
        timerSeconds = 0;
        updateTimerDisplay();
    }

    function updateTimerDisplay() {
        const m = Math.floor(timerSeconds / 60).toString().padStart(2, '0');
        const s = (timerSeconds % 60).toString().padStart(2, '0');
        document.getElementById('timer').textContent = `${m}:${s}`;
    }

    // ===== Submit =====
    async function submitCode() {
        if (!currentProblem) {
            showToast('请先选择一道题目', 'error');
            return;
        }

        const code = editor.getValue().trim();
        if (!code || code === currentProblem.template.trim()) {
            showToast('请先编写你的代码实现', 'error');
            return;
        }

        // Save code
        const pData = getProblemData(currentProblem.id);
        saveProblemData(currentProblem.id, { code, attempts: (pData.attempts || 0) + 1 });

        // Show loading
        document.getElementById('loadingOverlay').classList.add('visible');
        document.getElementById('submitCode').disabled = true;

        try {
            const result = await window.API.reviewCode(currentProblem, code);

            if (result.success) {
                displayReview(result.content);
                const score = extractScore(result.content);
                if (score !== null) {
                    const best = Math.max(score, pData.bestScore || 0);
                    saveProblemData(currentProblem.id, {
                        bestScore: best,
                        lastAttempt: new Date().toISOString(),
                        reviews: [...(pData.reviews || []).slice(-4), {
                            date: new Date().toISOString(),
                            score,
                            content: result.content
                        }]
                    });
                    renderProblemList();
                }
            } else {
                displayReview(`## API 调用失败\n\n${result.content}\n\n如果遇到 CORS 问题，可以尝试使用浏览器插件（如 CORS Unblock）或者本地代理服务。`);
            }
        } catch (err) {
            displayReview(`## 请求失败\n\n\`${err.message}\`\n\n可能的原因:\n- 网络连接问题\n- CORS 跨域限制\n- API 服务暂不可用\n\n**提示**: 如果遇到 CORS 问题，可以安装浏览器 CORS 插件或使用本地代理。`);
        } finally {
            document.getElementById('loadingOverlay').classList.remove('visible');
            document.getElementById('submitCode').disabled = false;
        }
    }

    function extractScore(content) {
        const patterns = [
            /总分[：:]\s*(\d+)\s*[/／]\s*100/,
            /(\d+)\s*[/／]\s*100\s*分/,
            /(\d+)\s*[/／]\s*100/,
            /总分[：:]\s*(\d+)/
        ];
        for (const pat of patterns) {
            const m = content.match(pat);
            if (m) return parseInt(m[1], 10);
        }
        return null;
    }

    function displayReview(content) {
        const body = document.getElementById('reviewBody');
        const html = marked.parse(content);

        const score = extractScore(content);
        let scoreBadge = '';
        if (score !== null) {
            const cls = score >= 80 ? 'score-high' : score >= 50 ? 'score-mid' : 'score-low';
            scoreBadge = `<div class="score-badge ${cls}">${score}/100</div>`;
        }

        body.innerHTML = `${scoreBadge}<div class="review-content">${html}</div>`;
    }

    function resetReview() {
        document.getElementById('reviewBody').innerHTML = `
            <div class="review-placeholder">
                <div class="placeholder-icon">🔍</div>
                <p>提交代码后，AI 将会对你的实现进行评审</p>
                <div class="review-features">
                    <div class="feature"><span>✅</span> 正确性检查</div>
                    <div class="feature"><span>📐</span> 维度/形状验证</div>
                    <div class="feature"><span>💡</span> 改进建议</div>
                    <div class="feature"><span>📊</span> 评分打分</div>
                </div>
            </div>
        `;
    }

    // ===== Modal =====
    function showModal(title, content) {
        document.getElementById('modalTitle').textContent = title;
        document.getElementById('modalBody').innerHTML = typeof content === 'string'
            ? marked.parse(content)
            : content;
        document.getElementById('modalOverlay').classList.add('visible');
    }

    function closeModal() {
        document.getElementById('modalOverlay').classList.remove('visible');
    }

    // ===== Theme =====
    function toggleTheme() {
        const body = document.body;
        const isLight = body.classList.toggle('light-theme');
        document.getElementById('toggleTheme').textContent = isLight ? '☀️' : '🌙';
        localStorage.setItem('algo_theme', isLight ? 'light' : 'dark');

        if (editor) {
            editor.setOption('theme', isLight ? 'default' : 'dracula');
        }
    }

    function restoreTheme() {
        const saved = localStorage.getItem('algo_theme');
        if (saved === 'light') {
            document.body.classList.add('light-theme');
            document.getElementById('toggleTheme').textContent = '☀️';
            if (editor) editor.setOption('theme', 'default');
        }
    }

    // ===== Views =====
    function switchView(view) {
        currentView = view;

        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.view === view);
        });

        // Toggle main content areas
        const sidebar = document.getElementById('sidebar');
        const workspace = document.querySelector('.workspace');
        const reviewPanel = document.getElementById('reviewPanel');

        // Remove any existing view containers
        document.querySelectorAll('.view-container').forEach(el => el.remove());

        if (view === 'practice') {
            sidebar.style.display = '';
            workspace.style.display = '';
            reviewPanel.style.display = '';
        } else if (view === 'progress') {
            sidebar.style.display = 'none';
            workspace.style.display = 'none';
            reviewPanel.style.display = 'none';
            showProgressView();
        } else if (view === 'review') {
            sidebar.style.display = 'none';
            workspace.style.display = 'none';
            reviewPanel.style.display = 'none';
            showReviewNotesView();
        }
    }

    function showProgressView() {
        const data = loadData();
        const total = window.PROBLEMS.length;
        let attempted = 0, completed = 0, totalScore = 0, scoredCount = 0;

        window.PROBLEMS.forEach(p => {
            const d = data[p.id];
            if (d && d.attempts > 0) attempted++;
            if (d && d.bestScore >= 70) completed++;
            if (d && d.bestScore) { totalScore += d.bestScore; scoredCount++; }
        });

        const avgScore = scoredCount > 0 ? Math.round(totalScore / scoredCount) : 0;

        const container = document.createElement('div');
        container.className = 'view-container active';
        container.innerHTML = `
            <h2 style="margin-bottom: 24px; font-size: 20px;">进度统计</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">${total}</div>
                    <div class="stat-label">总题数</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${attempted}</div>
                    <div class="stat-label">已尝试</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${completed}</div>
                    <div class="stat-label">已通过 (≥70分)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${avgScore}</div>
                    <div class="stat-label">平均分</div>
                </div>
            </div>
            <table class="progress-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>题目</th>
                        <th>难度</th>
                        <th>尝试次数</th>
                        <th>最高分</th>
                        <th>状态</th>
                    </tr>
                </thead>
                <tbody>
                    ${window.PROBLEMS.map(p => {
                        const d = data[p.id] || {};
                        const diffLabel = { easy: '基础', medium: '中等', hard: '困难' }[p.difficulty];
                        let statusCls = 'not-started', statusText = '未开始';
                        if (d.bestScore >= 70) { statusCls = 'completed'; statusText = '已通过'; }
                        else if (d.attempts > 0) { statusCls = 'attempted'; statusText = '进行中'; }

                        return `<tr>
                            <td>${p.id}</td>
                            <td>${p.title}</td>
                            <td><span class="difficulty-dot ${p.difficulty}"></span> ${diffLabel}</td>
                            <td>${d.attempts || 0}</td>
                            <td>${d.bestScore ? d.bestScore + '分' : '-'}</td>
                            <td><span class="status-badge ${statusCls}">${statusText}</span></td>
                        </tr>`;
                    }).join('')}
                </tbody>
            </table>
        `;
        document.querySelector('.main-content').appendChild(container);
    }

    function showReviewNotesView() {
        const data = loadData();
        const container = document.createElement('div');
        container.className = 'view-container active';

        let hasNotes = false;
        let notesHtml = '<div class="notes-list">';

        window.PROBLEMS.forEach(p => {
            const d = data[p.id];
            if (d && d.reviews && d.reviews.length > 0) {
                hasNotes = true;
                const latest = d.reviews[d.reviews.length - 1];
                const date = new Date(latest.date).toLocaleString('zh-CN');
                notesHtml += `
                    <div class="note-card">
                        <h4>${p.title}</h4>
                        <div class="note-meta">最近评审: ${date} | 得分: ${latest.score || '?'}分 | 尝试: ${d.attempts}次</div>
                        <div class="note-content">${marked.parse(latest.content.substring(0, 500) + (latest.content.length > 500 ? '...' : ''))}</div>
                    </div>
                `;
            }
        });

        notesHtml += '</div>';

        if (!hasNotes) {
            notesHtml = `
                <div class="empty-state">
                    <div class="empty-icon">📝</div>
                    <p>还没有评审记录。去做几道题吧！</p>
                </div>
            `;
        }

        container.innerHTML = `
            <h2 style="margin-bottom: 24px; font-size: 20px;">复习笔记</h2>
            ${notesHtml}
        `;
        document.querySelector('.main-content').appendChild(container);
    }

    // ===== Toast =====
    function showToast(msg, type) {
        const existing = document.querySelector('.toast');
        if (existing) existing.remove();

        const toast = document.createElement('div');
        toast.className = `toast ${type || 'info'}`;
        toast.textContent = msg;
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 3000);
    }

    // ===== Events =====
    function bindEvents() {
        // Problem list click
        document.getElementById('problemList').addEventListener('click', e => {
            const item = e.target.closest('.problem-item');
            if (item) selectProblem(parseInt(item.dataset.id, 10));
        });

        // Filters
        document.getElementById('categoryFilter').addEventListener('change', () => renderProblemList());
        document.getElementById('difficultyFilter').addEventListener('change', () => renderProblemList());

        // Nav tabs
        document.querySelectorAll('.nav-btn[data-view]').forEach(btn => {
            btn.addEventListener('click', () => switchView(btn.dataset.view));
        });

        // Theme
        document.getElementById('toggleTheme').addEventListener('click', toggleTheme);

        // Submit
        document.getElementById('submitCode').addEventListener('click', submitCode);

        // Reset code
        document.getElementById('resetCode').addEventListener('click', () => {
            if (!currentProblem) return;
            if (confirm('确定要重置代码吗？')) {
                editor.setValue(currentProblem.template);
                editor.clearHistory();
                saveProblemData(currentProblem.id, { code: null });
            }
        });

        // Format (basic indent)
        document.getElementById('formatCode').addEventListener('click', () => {
            if (!editor) return;
            const totalLines = editor.lineCount();
            for (let i = 0; i < totalLines; i++) {
                editor.indentLine(i, 'smart');
            }
            showToast('代码已格式化', 'success');
        });

        // Toggle description
        document.getElementById('toggleDescription').addEventListener('click', () => {
            document.getElementById('problemPanel').classList.toggle('collapsed');
        });

        // Show hint
        document.getElementById('showHint').addEventListener('click', () => {
            if (!currentProblem) return;
            const hints = currentProblem.hints;
            const content = hints.map((h, i) => `${i + 1}. ${h}`).join('\n');
            showModal('💡 提示', content);
        });

        // Show reference
        document.getElementById('showReference').addEventListener('click', () => {
            if (!currentProblem) return;
            showModal('📖 参考答案', '```python\n' + currentProblem.reference + '\n```');
        });

        // Modal close
        document.getElementById('modalClose').addEventListener('click', closeModal);
        document.getElementById('modalOverlay').addEventListener('click', e => {
            if (e.target === e.currentTarget) closeModal();
        });

        // Clear review
        document.getElementById('clearReview').addEventListener('click', resetReview);

        // Collapse / expand review panel
        document.getElementById('collapseReview').addEventListener('click', () => {
            document.getElementById('reviewPanel').classList.add('collapsed');
            setTimeout(() => editor && editor.refresh(), 320);
        });
        document.getElementById('reviewTab').addEventListener('click', () => {
            document.getElementById('reviewPanel').classList.remove('collapsed');
            setTimeout(() => editor && editor.refresh(), 320);
        });

        // ===== API Config Panel =====
        const configOverlay = document.getElementById('configOverlay');
        const cfgBaseUrl = document.getElementById('cfgBaseUrl');
        const cfgApiKey = document.getElementById('cfgApiKey');
        const cfgModel = document.getElementById('cfgModel');
        const cfgStatus = document.getElementById('cfgStatus');

        function openConfig() {
            const cfg = window.API.getConfig();
            cfgBaseUrl.value = cfg.baseUrl || '';
            cfgApiKey.value = cfg.apiKey || '';
            cfgModel.value = cfg.model || '';
            document.querySelectorAll('.provider-btn').forEach(b => {
                b.classList.toggle('active', b.dataset.provider === cfg.provider);
            });
            cfgStatus.className = 'config-status';
            cfgStatus.textContent = '';
            configOverlay.classList.add('visible');
        }

        function closeConfig() {
            configOverlay.classList.remove('visible');
        }

        function getFormConfig() {
            const activeProvider = document.querySelector('.provider-btn.active');
            return {
                provider: activeProvider ? activeProvider.dataset.provider : 'siliconflow',
                baseUrl: cfgBaseUrl.value.trim(),
                apiKey: cfgApiKey.value.trim(),
                model: cfgModel.value.trim()
            };
        }

        document.getElementById('openApiConfig').addEventListener('click', openConfig);
        document.getElementById('configClose').addEventListener('click', closeConfig);
        configOverlay.addEventListener('click', e => {
            if (e.target === configOverlay) closeConfig();
        });

        // Provider tabs
        document.querySelectorAll('.provider-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.provider-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                const provider = btn.dataset.provider;
                if (provider === 'jdcloud') {
                    cfgBaseUrl.value = 'https://modelservice.jdcloud.com/v1/responses';
                } else {
                    cfgBaseUrl.value = 'https://api.siliconflow.cn/v1/chat/completions';
                }
            });
        });

        // Presets
        document.querySelectorAll('.preset-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const preset = window.API.PRESETS[btn.dataset.preset];
                if (!preset) return;
                cfgBaseUrl.value = preset.baseUrl;
                if (preset.apiKey) cfgApiKey.value = preset.apiKey;
                cfgModel.value = preset.model;
                document.querySelectorAll('.provider-btn').forEach(b => {
                    b.classList.toggle('active', b.dataset.provider === preset.provider);
                });
            });
        });

        // Toggle key visibility
        document.getElementById('toggleKeyVisibility').addEventListener('click', () => {
            const input = cfgApiKey;
            input.type = input.type === 'password' ? 'text' : 'password';
        });

        // Test connection
        document.getElementById('cfgTest').addEventListener('click', async () => {
            const cfg = getFormConfig();
            if (!cfg.apiKey) {
                cfgStatus.className = 'config-status visible error';
                cfgStatus.textContent = '请填写 API Key';
                return;
            }
            cfgStatus.className = 'config-status visible';
            cfgStatus.style.color = 'var(--text-muted)';
            cfgStatus.style.background = 'var(--bg-tertiary)';
            cfgStatus.style.border = '1px solid var(--border-color)';
            cfgStatus.textContent = '测试中...';

            const result = await window.API.testConnection(cfg);
            cfgStatus.style = '';
            if (result.ok) {
                cfgStatus.className = 'config-status visible success';
                cfgStatus.textContent = result.msg;
            } else {
                cfgStatus.className = 'config-status visible error';
                cfgStatus.textContent = result.msg;
            }
        });

        // Save config
        document.getElementById('cfgSave').addEventListener('click', () => {
            const cfg = getFormConfig();
            if (!cfg.apiKey) {
                cfgStatus.className = 'config-status visible error';
                cfgStatus.textContent = '请填写 API Key';
                return;
            }
            window.API.saveConfig(cfg);
            showToast('API 配置已保存', 'success');
            closeConfig();
        });

        // Auto-save code
        let saveTimeout = null;
        if (editor) {
            editor.on('change', () => {
                if (!currentProblem) return;
                clearTimeout(saveTimeout);
                saveTimeout = setTimeout(() => {
                    saveProblemData(currentProblem.id, { code: editor.getValue() });
                }, 1000);
            });
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', e => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                submitCode();
            }
            if (e.key === 'Escape') {
                closeModal();
                closeConfig();
            }
        });
    }

    // ===== Start =====
    document.addEventListener('DOMContentLoaded', init);
})();
