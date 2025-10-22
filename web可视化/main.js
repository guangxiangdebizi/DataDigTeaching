// 主控制逻辑

// ============= 防抖函数 =============
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func.apply(this, args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// 数据生成函数
function generateData(type, numPoints = 200) {
    const points = [];
    
    switch (type) {
        case 'blobs':
            // 生成3个球形簇
            const centers = [
                {x: 0.3, y: 0.3},
                {x: 0.7, y: 0.7},
                {x: 0.5, y: 0.7}
            ];
            const pointsPerCluster = Math.floor(numPoints / 3);
            
            for (const center of centers) {
                for (let i = 0; i < pointsPerCluster; i++) {
                    const angle = Math.random() * 2 * Math.PI;
                    const r = Math.random() * 0.1;
                    points.push({
                        x: center.x + r * Math.cos(angle),
                        y: center.y + r * Math.sin(angle)
                    });
                }
            }
            break;
            
        case 'moons':
            // 生成月牙形
            const pointsPerMoon = Math.floor(numPoints / 2);
            
            for (let i = 0; i < pointsPerMoon; i++) {
                const angle = Math.PI * i / pointsPerMoon;
                const r = 0.3;
                const noise = (Math.random() - 0.5) * 0.05;
                points.push({
                    x: 0.5 + r * Math.cos(angle) + noise,
                    y: 0.4 + r * Math.sin(angle) + noise
                });
            }
            
            for (let i = 0; i < pointsPerMoon; i++) {
                const angle = Math.PI + Math.PI * i / pointsPerMoon;
                const r = 0.3;
                const noise = (Math.random() - 0.5) * 0.05;
                points.push({
                    x: 0.5 + r * Math.cos(angle) + noise,
                    y: 0.6 - r * Math.sin(angle) + noise
                });
            }
            break;
            
        case 'circles':
            // 生成同心圆
            const innerPoints = Math.floor(numPoints / 3);
            const outerPoints = numPoints - innerPoints;
            
            // 内圆
            for (let i = 0; i < innerPoints; i++) {
                const angle = Math.random() * 2 * Math.PI;
                const r = 0.15 + Math.random() * 0.05;
                points.push({
                    x: 0.5 + r * Math.cos(angle),
                    y: 0.5 + r * Math.sin(angle)
                });
            }
            
            // 外圆
            for (let i = 0; i < outerPoints; i++) {
                const angle = Math.random() * 2 * Math.PI;
                const r = 0.35 + Math.random() * 0.05;
                points.push({
                    x: 0.5 + r * Math.cos(angle),
                    y: 0.5 + r * Math.sin(angle)
                });
            }
            break;
            
        case 'random':
            // 随机点
            for (let i = 0; i < numPoints; i++) {
                points.push({
                    x: Math.random() * 0.8 + 0.1,
                    y: Math.random() * 0.8 + 0.1
                });
            }
            break;
    }
    
    return points;
}

// 主应用类
class ClusteringApp {
    constructor() {
        this.canvas = document.getElementById('mainCanvas');
        this.visualizer = new Visualizer(this.canvas);
        
        this.currentAlgorithm = 'dbscan';
        this.currentDataset = 'blobs';
        this.points = [];
        this.steps = [];
        this.currentStep = 0;
        this.isPlaying = false;
        this.animationInterval = null;
        this.speed = 1000; // 毫秒
        
        this.customMode = false;
        
        this.init();
    }
    
    init() {
        // 生成初始数据
        this.generateNewData();
        
        // 绑定事件
        this.bindEvents();
        
        // 设置Canvas自适应
        this.setupCanvasResize();
        
        // 初始化手风琴
        this.initAccordion();
        
        // 初始化帮助模态框
        this.initHelpModal();
        
        // 初始化移动端操作栏
        this.initMobileControls();
        
        // 初始化步骤指示器折叠功能
        this.initStepToggle();
        
        // 运行初始聚类
        this.runClustering();
    }
    
    // ============= Canvas 自适应 =============
    setupCanvasResize() {
        const resizeCanvas = () => {
            const wrapper = this.canvas.parentElement;
            const rect = wrapper.getBoundingClientRect();
            const dpr = window.devicePixelRatio || 1;
            
            // 设置实际分辨率（高DPI）
            this.canvas.width = rect.width * dpr;
            this.canvas.height = rect.height * dpr;
            
            // 缩放上下文
            this.visualizer.ctx.scale(dpr, dpr);
            
            // 更新可视化器的尺寸
            this.visualizer.width = rect.width;
            this.visualizer.height = rect.height;
            
            // 重绘当前内容
            this.drawCurrentStep();
        };
        
        window.addEventListener('resize', debounce(resizeCanvas, 250));
        resizeCanvas(); // 初始化
    }
    
    // ============= 手风琴交互 =============
    initAccordion() {
        document.querySelectorAll('.accordion-header').forEach((header, index) => {
            header.addEventListener('click', () => {
                const content = header.nextElementSibling;
                const icon = header.querySelector('.accordion-icon');
                const isActive = content.classList.contains('show');
                
                // 移动端：一次只展开一个
                if (window.innerWidth < 768) {
                    document.querySelectorAll('.accordion-content').forEach(c => {
                        c.classList.remove('show');
                    });
                    document.querySelectorAll('.accordion-header').forEach(h => {
                        h.classList.remove('active');
                        const i = h.querySelector('.accordion-icon');
                        if (i.textContent === '▼') {
                            i.textContent = '▶';
                        }
                    });
                }
                
                if (!isActive) {
                    content.classList.add('show');
                    header.classList.add('active');
                    icon.textContent = '▼';
                } else {
                    content.classList.remove('show');
                    header.classList.remove('active');
                    icon.textContent = '▶';
                }
            });
        });
    }
    
    // ============= 帮助模态框 =============
    initHelpModal() {
        const modal = document.getElementById('helpModal');
        const helpBtn = document.getElementById('helpBtn');
        const closeBtn = modal.querySelector('.modal-close');
        const overlay = modal.querySelector('.modal-overlay');
        
        helpBtn.addEventListener('click', () => {
            modal.classList.add('show');
            document.body.style.overflow = 'hidden';
        });
        
        const closeModal = () => {
            modal.classList.remove('show');
            document.body.style.overflow = 'auto';
        };
        
        closeBtn.addEventListener('click', closeModal);
        overlay.addEventListener('click', closeModal);
        
        // ESC 关闭
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && modal.classList.contains('show')) {
                closeModal();
            }
        });
    }
    
    // ============= 移动端操作栏 =============
    initMobileControls() {
        document.getElementById('playBtnMobile').addEventListener('click', () => this.play());
        document.getElementById('pauseBtnMobile').addEventListener('click', () => this.pause());
        document.getElementById('resetBtnMobile').addEventListener('click', () => this.reset());
        document.getElementById('stepBtnMobile').addEventListener('click', () => this.nextStep());
    }
    
    // ============= 步骤指示器折叠 =============
    initStepToggle() {
        const stepToggle = document.getElementById('stepToggle');
        const stepIndicator = document.querySelector('.step-indicator');
        const toggleText = document.querySelector('.toggle-text');
        
        stepToggle.addEventListener('click', () => {
            const isCollapsed = stepIndicator.classList.contains('collapsed');
            
            if (isCollapsed) {
                stepIndicator.classList.remove('collapsed');
                toggleText.textContent = '收起';
            } else {
                stepIndicator.classList.add('collapsed');
                toggleText.textContent = '步骤';
            }
        });
        
        // 允许点击按钮（移除pointer-events: none的限制）
        stepToggle.style.pointerEvents = 'auto';
    }
    
    bindEvents() {
        // 算法选择
        document.querySelectorAll('.algo-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.algo-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.currentAlgorithm = btn.dataset.algo;
                this.switchAlgorithm();
            });
        });
        
        // 数据集选择
        document.getElementById('datasetSelect').addEventListener('change', (e) => {
            this.currentDataset = e.target.value;
            this.customMode = (e.target.value === 'custom');
            if (!this.customMode) {
                this.generateNewData();
                this.runClustering();
            } else {
                this.points = [];
                this.steps = [];
                this.currentStep = 0;
                this.visualizer.clear();
                this.updateStats();
            }
        });
        
        document.getElementById('generateData').addEventListener('click', () => {
            if (!this.customMode) {
                this.generateNewData();
                this.runClustering();
            }
        });
        
        // 画布点击（自定义模式）
        this.canvas.addEventListener('click', (e) => {
            if (this.customMode) {
                const rect = this.canvas.getBoundingClientRect();
                const x = (e.clientX - rect.left) / rect.width;
                const y = (e.clientY - rect.top) / rect.height;
                
                this.points.push({x, y});
                this.runClustering();
            }
        });
        
        // DBSCAN参数 - 使用防抖
        document.getElementById('epsSlider').addEventListener('input', debounce((e) => {
            document.getElementById('epsValue').textContent = e.target.value;
            if (this.currentAlgorithm === 'dbscan') {
                this.runClustering();
            }
        }, 500));
        
        document.getElementById('minPtsSlider').addEventListener('input', debounce((e) => {
            document.getElementById('minPtsValue').textContent = e.target.value;
            if (this.currentAlgorithm === 'dbscan') {
                this.runClustering();
            }
        }, 500));
        
        // 网格参数 - 使用防抖
        document.getElementById('gridSizeSlider').addEventListener('input', debounce((e) => {
            document.getElementById('gridSizeValue').textContent = e.target.value;
            if (this.currentAlgorithm === 'grid') {
                this.runClustering();
            }
        }, 500));
        
        document.getElementById('densitySlider').addEventListener('input', debounce((e) => {
            document.getElementById('densityValue').textContent = e.target.value;
            if (this.currentAlgorithm === 'grid') {
                this.runClustering();
            }
        }, 500));
        
        // GMM参数
        document.getElementById('kSlider').addEventListener('input', debounce((e) => {
            document.getElementById('kValue').textContent = e.target.value;
            if (this.currentAlgorithm === 'gmm') {
                this.runClustering();
            }
        }, 500));
        
        document.getElementById('showEllipse').addEventListener('change', (e) => {
            if (this.currentAlgorithm === 'gmm') {
                this.drawCurrentStep();
            }
        });
        
        // 动画控制
        document.getElementById('playBtn').addEventListener('click', () => this.play());
        document.getElementById('pauseBtn').addEventListener('click', () => this.pause());
        document.getElementById('resetBtn').addEventListener('click', () => this.reset());
        document.getElementById('stepBtn').addEventListener('click', () => this.nextStep());
        
        document.getElementById('speedSlider').addEventListener('input', (e) => {
            const speeds = ['很慢', '慢', '正常', '快', '很快'];
            const speedValues = [2000, 1500, 1000, 500, 200];
            const index = parseInt(e.target.value) - 1;
            document.getElementById('speedValue').textContent = speeds[index];
            this.speed = speedValues[index];
            
            if (this.isPlaying) {
                this.pause();
                this.play();
            }
        });
    }
    
    switchAlgorithm() {
        // 隐藏所有参数面板
        document.getElementById('dbscanParams').classList.add('hidden');
        document.getElementById('gridParams').classList.add('hidden');
        document.getElementById('gmmParams').classList.add('hidden');
        
        // 显示当前算法的参数面板
        switch (this.currentAlgorithm) {
            case 'dbscan':
                document.getElementById('dbscanParams').classList.remove('hidden');
                this.updateAlgorithmInfo('DBSCAN（基于密度的聚类）', [
                    '🔵 蓝色圆圈：邻域范围',
                    '🟢 绿色填充：已分配的点',
                    '🔴 红色：噪声点',
                    '⚫ 黑色边框：核心点',
                    '⚡ 可发现任意形状的簇'
                ]);
                break;
            case 'grid':
                document.getElementById('gridParams').classList.remove('hidden');
                this.updateAlgorithmInfo('网格聚类（CLIQUE）', [
                    '📊 网格划分数据空间',
                    '🟦 蓝色网格：密集单元',
                    '🟡 黄色高亮：当前处理的网格',
                    '🔗 连接相邻密集网格形成簇',
                    '⚡ 时间复杂度O(n)，快速处理'
                ]);
                break;
            case 'gmm':
                document.getElementById('gmmParams').classList.remove('hidden');
                this.updateAlgorithmInfo('GMM（高斯混合模型）', [
                    '🟣 椭圆：高斯分布（2σ范围）',
                    '✖️ 十字标记：均值中心',
                    '🎨 点透明度：归属概率',
                    '🔄 EM算法迭代优化',
                    '⚡ 软聚类，输出概率'
                ]);
                break;
        }
        
        this.runClustering();
    }
    
    updateAlgorithmInfo(title, items) {
        const infoBox = document.getElementById('algoInfo');
        let html = `<p><strong>${title}</strong></p><ul>`;
        for (const item of items) {
            html += `<li>${item}</li>`;
        }
        html += '</ul>';
        infoBox.innerHTML = html;
    }
    
    generateNewData() {
        this.points = generateData(this.currentDataset, 200);
    }
    
    runClustering() {
        if (this.points.length === 0) {
            this.steps = [];
            this.currentStep = 0;
            this.updateStats();
            return;
        }
        
        this.pause();
        let result;
        
        switch (this.currentAlgorithm) {
            case 'dbscan':
                const eps = parseFloat(document.getElementById('epsSlider').value);
                const minPts = parseInt(document.getElementById('minPtsSlider').value);
                const dbscan = new DBSCAN(eps, minPts);
                result = dbscan.fit(this.points);
                break;
                
            case 'grid':
                const gridSize = parseInt(document.getElementById('gridSizeSlider').value);
                const density = parseInt(document.getElementById('densitySlider').value);
                const grid = new GridClustering(gridSize, density);
                result = grid.fit(this.points);
                break;
                
            case 'gmm':
                const k = parseInt(document.getElementById('kSlider').value);
                const gmm = new GMM(k);
                result = gmm.fit(this.points);
                break;
        }
        
        this.steps = result.steps;
        this.currentStep = 0;
        this.drawCurrentStep();
        this.updateStats();
    }
    
    drawCurrentStep() {
        if (this.steps.length === 0) {
            this.visualizer.clear();
            return;
        }
        
        const step = this.steps[this.currentStep];
        const options = {};
        
        if (this.currentAlgorithm === 'gmm') {
            options.showEllipse = document.getElementById('showEllipse').checked;
        }
        
        this.visualizer.draw(step, this.currentAlgorithm, options);
        this.visualizer.updateLegend(step, this.currentAlgorithm);
        
        // 更新步骤描述
        document.getElementById('stepDescription').textContent = step.description || '';
        
        // 更新进度条
        const progress = (this.currentStep / Math.max(1, this.steps.length - 1)) * 100;
        document.getElementById('progressFill').style.width = progress + '%';
    }
    
    updateStats() {
        document.getElementById('pointCount').textContent = this.points.length;
        document.getElementById('currentStep').textContent = 
            `${this.currentStep + 1} / ${this.steps.length}`;
        
        if (this.steps.length > 0) {
            const step = this.steps[this.currentStep];
            const labels = step.points.map(p => p.label);
            const numClusters = new Set(labels.filter(l => l >= 0)).size;
            const numNoise = labels.filter(l => l === -1).length;
            
            document.getElementById('clusterCount').textContent = numClusters;
            document.getElementById('noiseCount').textContent = numNoise;
        } else {
            document.getElementById('clusterCount').textContent = 0;
            document.getElementById('noiseCount').textContent = 0;
        }
    }
    
    play() {
        if (this.steps.length === 0) return;
        
        this.isPlaying = true;
        document.getElementById('playBtn').disabled = true;
        
        this.animationInterval = setInterval(() => {
            if (this.currentStep < this.steps.length - 1) {
                this.currentStep++;
                this.drawCurrentStep();
                this.updateStats();
            } else {
                this.pause();
            }
        }, this.speed);
    }
    
    pause() {
        this.isPlaying = false;
        document.getElementById('playBtn').disabled = false;
        if (this.animationInterval) {
            clearInterval(this.animationInterval);
            this.animationInterval = null;
        }
    }
    
    reset() {
        this.pause();
        this.currentStep = 0;
        this.drawCurrentStep();
        this.updateStats();
    }
    
    nextStep() {
        if (this.steps.length === 0) return;
        
        this.pause();
        if (this.currentStep < this.steps.length - 1) {
            this.currentStep++;
            this.drawCurrentStep();
            this.updateStats();
        }
    }
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    const app = new ClusteringApp();
    
    // 添加欢迎提示
    setTimeout(() => {
        console.log('%c🎉 聚类算法可视化系统已启动！', 'color: #667eea; font-size: 16px; font-weight: bold;');
        console.log('%c💡 提示：尝试不同的算法和数据集，观察聚类过程！', 'color: #4ECDC4; font-size: 14px;');
    }, 100);
});

