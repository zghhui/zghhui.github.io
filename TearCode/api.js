window.API = {
    STORAGE_KEY: 'algo_api_config',

    PRESETS: {
        'siliconflow-qwen': {
            provider: 'siliconflow',
            baseUrl: 'https://api.siliconflow.cn/v1/chat/completions',
            apiKey: '',
            model: 'Qwen/Qwen3-8B'
        },
        'siliconflow-deepseek': {
            provider: 'siliconflow',
            baseUrl: 'https://api.siliconflow.cn/v1/chat/completions',
            apiKey: '',
            model: 'deepseek-ai/DeepSeek-V4-Flash'
        }
    },

    DEFAULT_CONFIG: {
        provider: 'siliconflow',
        baseUrl: 'https://api.siliconflow.cn/v1/chat/completions',
        apiKey: '',
        model: 'deepseek-ai/DeepSeek-V4-Flash'
    },

    getConfig() {
        try {
            const saved = JSON.parse(localStorage.getItem(this.STORAGE_KEY));
            return saved || { ...this.DEFAULT_CONFIG };
        } catch { return { ...this.DEFAULT_CONFIG }; }
    },

    saveConfig(cfg) {
        localStorage.setItem(this.STORAGE_KEY, JSON.stringify(cfg));
    },

    _buildPrompt(problem, userCode) {
        return `你是一位深度学习算法面试官，正在面试候选人的"手撕代码"能力。
请仔细评审以下代码实现。

## 题目
${problem.title}

## 题目要求
${problem.description}

## 候选人代码
\`\`\`python
${userCode}
\`\`\`

## 参考答案
\`\`\`python
${problem.reference}
\`\`\`

## 评审要求
请从以下几个维度评审，并给出总分（满分100分）：

### 1. 正确性 (40分)
- 核心逻辑是否正确
- 关键公式是否正确实现
- 边界条件处理

### 2. 代码质量 (20分)
- 代码结构清晰度
- 变量命名规范性
- 是否有冗余代码

### 3. 关键细节 (25分)
- 维度变换是否正确 (view/reshape/transpose)
- 数值稳定性处理 (eps, clamp 等)
- 梯度相关处理 (no_grad, detach 等)

### 4. 完整度 (15分)
- 是否实现了所有要求的功能
- 是否有遗漏的关键步骤

## 输出格式
请用 Markdown 格式输出，包含：
1. **总分**: X/100
2. **正确性分析**: 逐行检查关键实现
3. **问题列表**: 具体指出哪里有错（如果有）
4. **改进建议**: 如何提升代码质量
5. **总体评价**: 一句话总结

注意：请用中文回答，评价要具体、有建设性。如果代码基本为空或只有注释，请明确指出需要补充实现。`;
    },

    async reviewCode(problem, userCode) {
        const cfg = this.getConfig();
        if (!cfg.apiKey) {
            return { success: false, content: '请先在 API 配置中填写 API Key（点击顶部 ⚙️ 按钮）' };
        }
        const prompt = this._buildPrompt(problem, userCode);

        if (cfg.provider === 'jdcloud') {
            return this._callJDCloud(cfg, prompt);
        } else {
            return this._callOpenAICompat(cfg, prompt);
        }
    },

    async _callOpenAICompat(cfg, prompt) {
        const payload = {
            model: cfg.model,
            messages: [
                { role: 'system', content: '你是一位深度学习算法面试官，擅长评审 PyTorch 手撕代码。' },
                { role: 'user', content: prompt }
            ]
        };

        const response = await fetch(cfg.baseUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${cfg.apiKey}`
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (response.ok && data.choices && data.choices[0]) {
            return {
                success: true,
                content: data.choices[0].message.content
            };
        }
        return {
            success: false,
            content: `API 调用失败 (${response.status}): ${JSON.stringify(data)}`
        };
    },

    async _callJDCloud(cfg, prompt) {
        const payload = {
            model: cfg.model,
            stream: false,
            contents: {
                role: 'user',
                parts: [{ text: prompt }]
            }
        };

        const response = await fetch(cfg.baseUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${cfg.apiKey}`,
                'Trace-Id': this._uuid()
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (response.ok && data.candidates && data.candidates[0]) {
            return {
                success: true,
                content: data.candidates[0].content.parts[0].text
            };
        }
        return {
            success: false,
            content: `API 调用失败 (${response.status}): ${JSON.stringify(data)}`
        };
    },

    async testConnection(cfg) {
        try {
            if (cfg.provider === 'jdcloud') {
                const res = await this._callJDCloud(cfg, '请回复 ok');
                return res.success ? { ok: true, msg: '连接成功' } : { ok: false, msg: res.content };
            } else {
                const payload = {
                    model: cfg.model,
                    messages: [{ role: 'user', content: '请回复 ok' }],
                    max_tokens: 10
                };
                const response = await fetch(cfg.baseUrl, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${cfg.apiKey}`
                    },
                    body: JSON.stringify(payload)
                });
                const data = await response.json();
                if (response.ok && data.choices) {
                    return { ok: true, msg: `连接成功 (${cfg.model})` };
                }
                return { ok: false, msg: `${response.status}: ${data.error?.message || JSON.stringify(data)}` };
            }
        } catch (e) {
            return { ok: false, msg: e.message };
        }
    },

    _uuid() {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
            const r = Math.random() * 16 | 0;
            return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
        });
    }
};
