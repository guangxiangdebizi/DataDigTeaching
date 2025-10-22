// 可视化绘制模块

class Visualizer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.width = canvas.clientWidth;
        this.height = canvas.clientHeight;
        this.scale = 1;
        this.offsetX = 0;
        this.offsetY = 0;
        
        // 颜色方案
        this.colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
            '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739', '#52B788'
        ];
        
        this.noiseColor = '#FF0000';
        this.unvisitedColor = '#CCCCCC';
    }

    // 获取当前Canvas尺寸
    getCanvasSize() {
        return {
            width: this.width,
            height: this.height
        };
    }

    clear() {
        this.ctx.clearRect(0, 0, this.width, this.height);
        // 绘制背景网格
        this.drawGrid();
    }

    drawGrid() {
        this.ctx.strokeStyle = '#F0F0F0';
        this.ctx.lineWidth = 1;
        
        const gridSize = 50;
        for (let x = 0; x < this.width; x += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, this.height);
            this.ctx.stroke();
        }
        
        for (let y = 0; y < this.height; y += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.width, y);
            this.ctx.stroke();
        }
    }

    // 将数据坐标转换为画布坐标
    toCanvasCoords(x, y, bounds) {
        if (!bounds) {
            return {x: x * this.width, y: y * this.height};
        }
        
        const padding = 50;
        const usableWidth = this.width - 2 * padding;
        const usableHeight = this.height - 2 * padding;
        
        const scaleX = usableWidth / (bounds.maxX - bounds.minX);
        const scaleY = usableHeight / (bounds.maxY - bounds.minY);
        
        const canvasX = padding + (x - bounds.minX) * scaleX;
        const canvasY = this.height - padding - (y - bounds.minY) * scaleY;
        
        return {x: canvasX, y: canvasY};
    }

    // 计算数据边界
    getBounds(points) {
        if (points.length === 0) return {minX: 0, maxX: 1, minY: 0, maxY: 1};
        
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;
        
        for (const p of points) {
            minX = Math.min(minX, p.x);
            maxX = Math.max(maxX, p.x);
            minY = Math.min(minY, p.y);
            maxY = Math.max(maxY, p.y);
        }
        
        // 添加一些边距
        const marginX = (maxX - minX) * 0.1;
        const marginY = (maxY - minY) * 0.1;
        
        return {
            minX: minX - marginX,
            maxX: maxX + marginX,
            minY: minY - marginY,
            maxY: maxY + marginY
        };
    }

    // 绘制DBSCAN步骤
    drawDBSCANStep(step) {
        this.clear();
        const bounds = this.getBounds(step.points);
        
        // 绘制邻域圆（如果有eps参数）
        if (step.eps && step.type === 'find_core') {
            for (const p of step.points) {
                if (p.isCore) {
                    const pos = this.toCanvasCoords(p.x, p.y, bounds);
                    const radius = step.eps * (this.width - 100) / (bounds.maxX - bounds.minX);
                    
                    this.ctx.strokeStyle = 'rgba(102, 126, 234, 0.2)';
                    this.ctx.lineWidth = 2;
                    this.ctx.beginPath();
                    this.ctx.arc(pos.x, pos.y, radius, 0, 2 * Math.PI);
                    this.ctx.stroke();
                }
            }
        }

        // 绘制当前点的邻域（高亮）
        if (step.type === 'expand' || step.type === 'new_cluster') {
            for (const p of step.points) {
                if (p.current) {
                    const pos = this.toCanvasCoords(p.x, p.y, bounds);
                    const radius = step.eps ? step.eps * (this.width - 100) / (bounds.maxX - bounds.minX) : 30;
                    
                    this.ctx.strokeStyle = 'rgba(255, 215, 0, 0.6)';
                    this.ctx.lineWidth = 3;
                    this.ctx.beginPath();
                    this.ctx.arc(pos.x, pos.y, radius, 0, 2 * Math.PI);
                    this.ctx.stroke();
                }
            }
        }
        
        // 绘制点
        for (const p of step.points) {
            const pos = this.toCanvasCoords(p.x, p.y, bounds);
            
            // 确定颜色
            let color;
            if (p.label === -2 || p.label === undefined) {
                color = this.unvisitedColor;
            } else if (p.label === -1) {
                color = this.noiseColor;
            } else {
                color = this.colors[p.label % this.colors.length];
            }
            
            // 绘制点
            this.ctx.fillStyle = color;
            this.ctx.beginPath();
            this.ctx.arc(pos.x, pos.y, p.current ? 8 : 6, 0, 2 * Math.PI);
            this.ctx.fill();
            
            // 核心点标记
            if (p.isCore) {
                this.ctx.strokeStyle = '#000';
                this.ctx.lineWidth = 2;
                this.ctx.beginPath();
                this.ctx.arc(pos.x, pos.y, 9, 0, 2 * Math.PI);
                this.ctx.stroke();
            }
            
            // 当前点高亮
            if (p.current) {
                this.ctx.strokeStyle = '#FFD700';
                this.ctx.lineWidth = 3;
                this.ctx.beginPath();
                this.ctx.arc(pos.x, pos.y, 12, 0, 2 * Math.PI);
                this.ctx.stroke();
            }
        }
    }

    // 绘制网格聚类步骤
    drawGridStep(step) {
        this.clear();
        const grid = step.grid;
        
        if (!grid) return;
        
        const padding = 50;
        const usableWidth = this.width - 2 * padding;
        const usableHeight = this.height - 2 * padding;
        
        const cellWidth = usableWidth / grid.size;
        const cellHeight = usableHeight / grid.size;
        
        // 绘制网格线
        this.ctx.strokeStyle = '#DDD';
        this.ctx.lineWidth = 1;
        
        for (let i = 0; i <= grid.size; i++) {
            const x = padding + i * cellWidth;
            this.ctx.beginPath();
            this.ctx.moveTo(x, padding);
            this.ctx.lineTo(x, this.height - padding);
            this.ctx.stroke();
            
            const y = padding + i * cellHeight;
            this.ctx.beginPath();
            this.ctx.moveTo(padding, y);
            this.ctx.lineTo(this.width - padding, y);
            this.ctx.stroke();
        }
        
        // 绘制密集网格
        if (step.denseGrids) {
            for (const key of step.denseGrids) {
                const [gridX, gridY] = key.split(',').map(Number);
                const x = padding + gridX * cellWidth;
                const y = padding + gridY * cellHeight;
                
                // 判断是否是当前簇的网格
                let isCurrent = false;
                if (step.currentGrid === key) {
                    isCurrent = true;
                }
                
                this.ctx.fillStyle = isCurrent ? 'rgba(255, 215, 0, 0.4)' : 'rgba(102, 126, 234, 0.2)';
                this.ctx.fillRect(x, y, cellWidth, cellHeight);
                
                this.ctx.strokeStyle = isCurrent ? '#FFD700' : '#667eea';
                this.ctx.lineWidth = 2;
                this.ctx.strokeRect(x, y, cellWidth, cellHeight);
            }
        }
        
        // 绘制点
        const bounds = {
            minX: grid.minX,
            maxX: grid.maxX,
            minY: grid.minY,
            maxY: grid.maxY
        };
        
        for (const p of step.points) {
            const pos = this.toCanvasCoords(p.x, p.y, bounds);
            
            let color;
            if (p.label === -1) {
                color = this.noiseColor;
            } else {
                color = this.colors[p.label % this.colors.length];
            }
            
            this.ctx.fillStyle = color;
            this.ctx.beginPath();
            this.ctx.arc(pos.x, pos.y, 5, 0, 2 * Math.PI);
            this.ctx.fill();
            
            this.ctx.strokeStyle = '#000';
            this.ctx.lineWidth = 1;
            this.ctx.stroke();
        }
    }

    // 绘制GMM步骤
    drawGMMStep(step, showEllipse = true) {
        this.clear();
        const bounds = this.getBounds(step.points);
        
        // 绘制高斯椭圆
        if (showEllipse && step.means && step.covariances) {
            for (let i = 0; i < step.means.length; i++) {
                this.drawGaussianEllipse(step.means[i], step.covariances[i], bounds, i);
            }
        }
        
        // 绘制点
        for (const p of step.points) {
            const pos = this.toCanvasCoords(p.x, p.y, bounds);
            
            // 如果有概率信息，使用概率着色
            let color;
            if (p.label !== undefined && p.label >= 0) {
                color = this.colors[p.label % this.colors.length];
                
                // 如果有概率信息，调整透明度
                if (p.probs && p.probs.length > 0) {
                    const maxProb = Math.max(...p.probs);
                    const alpha = 0.3 + 0.7 * maxProb;
                    const rgb = this.hexToRgb(color);
                    color = `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${alpha})`;
                }
            } else {
                color = this.unvisitedColor;
            }
            
            this.ctx.fillStyle = color;
            this.ctx.beginPath();
            this.ctx.arc(pos.x, pos.y, 5, 0, 2 * Math.PI);
            this.ctx.fill();
            
            this.ctx.strokeStyle = '#000';
            this.ctx.lineWidth = 1;
            this.ctx.stroke();
        }
        
        // 绘制均值点
        if (step.means) {
            for (let i = 0; i < step.means.length; i++) {
                const pos = this.toCanvasCoords(step.means[i].x, step.means[i].y, bounds);
                
                // 绘制十字标记
                this.ctx.strokeStyle = this.colors[i % this.colors.length];
                this.ctx.lineWidth = 3;
                this.ctx.beginPath();
                this.ctx.moveTo(pos.x - 10, pos.y);
                this.ctx.lineTo(pos.x + 10, pos.y);
                this.ctx.moveTo(pos.x, pos.y - 10);
                this.ctx.lineTo(pos.x, pos.y + 10);
                this.ctx.stroke();
                
                // 绘制圆圈
                this.ctx.strokeStyle = '#000';
                this.ctx.lineWidth = 2;
                this.ctx.beginPath();
                this.ctx.arc(pos.x, pos.y, 6, 0, 2 * Math.PI);
                this.ctx.stroke();
            }
        }
    }

    // 绘制高斯椭圆
    drawGaussianEllipse(mean, covariance, bounds, colorIndex) {
        const pos = this.toCanvasCoords(mean.x, mean.y, bounds);
        
        // 计算特征值和特征向量
        const a = covariance[0][0];
        const b = covariance[0][1];
        const c = covariance[1][0];
        const d = covariance[1][1];
        
        const trace = a + d;
        const det = a * d - b * c;
        
        const lambda1 = trace / 2 + Math.sqrt(trace * trace / 4 - det);
        const lambda2 = trace / 2 - Math.sqrt(trace * trace / 4 - det);
        
        // 计算旋转角度
        let angle = 0;
        if (Math.abs(b) > 1e-10) {
            angle = Math.atan2(lambda1 - a, b);
        } else if (a > d) {
            angle = 0;
        } else {
            angle = Math.PI / 2;
        }
        
        // 计算椭圆的轴长（2倍标准差）
        const scale = (this.width - 100) / (bounds.maxX - bounds.minX);
        const width = 2 * 2 * Math.sqrt(Math.abs(lambda1)) * scale;
        const height = 2 * 2 * Math.sqrt(Math.abs(lambda2)) * scale;
        
        // 绘制椭圆
        this.ctx.save();
        this.ctx.translate(pos.x, pos.y);
        this.ctx.rotate(angle);
        
        this.ctx.strokeStyle = this.colors[colorIndex % this.colors.length];
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([5, 5]);
        this.ctx.beginPath();
        this.ctx.ellipse(0, 0, width / 2, height / 2, 0, 0, 2 * Math.PI);
        this.ctx.stroke();
        this.ctx.setLineDash([]);
        
        this.ctx.restore();
    }

    // 辅助函数：十六进制颜色转RGB
    hexToRgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : {r: 0, g: 0, b: 0};
    }

    // 绘制统一接口
    draw(step, algorithm, options = {}) {
        if (!step) return;
        
        switch (algorithm) {
            case 'dbscan':
                this.drawDBSCANStep(step);
                break;
            case 'grid':
                this.drawGridStep(step);
                break;
            case 'gmm':
                this.drawGMMStep(step, options.showEllipse);
                break;
        }
    }

    // 更新图例
    updateLegend(step, algorithm) {
        const legendItems = document.getElementById('legendItems');
        legendItems.innerHTML = '';
        
        if (!step || !step.points) return;
        
        const labels = new Set(step.points.map(p => p.label).filter(l => l !== undefined));
        
        // 噪声点
        if (labels.has(-1)) {
            this.addLegendItem('噪声点', this.noiseColor);
        }
        
        // 未访问点
        if (labels.has(-2)) {
            this.addLegendItem('未访问', this.unvisitedColor);
        }
        
        // 各个簇
        const clusters = Array.from(labels).filter(l => l >= 0).sort((a, b) => a - b);
        for (const label of clusters) {
            const color = this.colors[label % this.colors.length];
            this.addLegendItem(`簇 ${label}`, color);
        }
        
        // DBSCAN特殊标记
        if (algorithm === 'dbscan') {
            this.addLegendItem('核心点（黑边）', '#000', true);
        }
        
        // GMM特殊标记
        if (algorithm === 'gmm') {
            this.addLegendItem('均值中心（十字）', '#000', false, true);
        }
    }

    addLegendItem(label, color, isBorder = false, isCross = false) {
        const legendItems = document.getElementById('legendItems');
        const item = document.createElement('div');
        item.className = 'legend-item';
        
        if (isCross) {
            item.innerHTML = `
                <div style="width: 20px; height: 20px; position: relative;">
                    <div style="position: absolute; width: 100%; height: 2px; top: 9px; background: ${color};"></div>
                    <div style="position: absolute; width: 2px; height: 100%; left: 9px; background: ${color};"></div>
                </div>
                <span>${label}</span>
            `;
        } else {
            const colorDiv = document.createElement('div');
            colorDiv.className = 'legend-color';
            colorDiv.style.background = color;
            if (isBorder) {
                colorDiv.style.border = `3px solid ${color}`;
                colorDiv.style.background = 'transparent';
            }
            
            const span = document.createElement('span');
            span.textContent = label;
            
            item.appendChild(colorDiv);
            item.appendChild(span);
        }
        
        legendItems.appendChild(item);
    }
}

