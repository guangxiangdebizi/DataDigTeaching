// 聚类算法实现
// 支持逐步执行以便动画展示

// ============================================================================
// DBSCAN 算法
// ============================================================================
class DBSCAN {
    constructor(eps, minPts) {
        this.eps = eps;
        this.minPts = minPts;
        this.steps = []; // 存储每一步的状态
    }

    fit(points) {
        this.steps = [];
        const n = points.length;
        const labels = new Array(n).fill(-2); // -2: 未访问, -1: 噪声, >=0: 簇ID
        const corePoints = new Set();
        let clusterId = 0;

        // 步骤1: 识别所有核心点
        this.steps.push({
            type: 'init',
            description: '初始化：所有点标记为未访问',
            points: points.map((p, i) => ({...p, label: -2, isCore: false}))
        });

        // 计算每个点的邻域
        const neighbors = [];
        for (let i = 0; i < n; i++) {
            const neighborhood = [];
            for (let j = 0; j < n; j++) {
                if (i !== j && this.distance(points[i], points[j]) <= this.eps) {
                    neighborhood.push(j);
                }
            }
            neighbors.push(neighborhood);
            
            if (neighborhood.length >= this.minPts - 1) {
                corePoints.add(i);
            }
        }

        this.steps.push({
            type: 'find_core',
            description: `找到 ${corePoints.size} 个核心点（邻域内至少有 ${this.minPts} 个点）`,
            points: points.map((p, i) => ({
                ...p, 
                label: -2, 
                isCore: corePoints.has(i),
                neighbors: neighbors[i]
            })),
            eps: this.eps
        });

        // 步骤2: 从核心点开始扩展簇
        for (let i = 0; i < n; i++) {
            if (labels[i] !== -2) continue;
            if (!corePoints.has(i)) continue;

            // 开始新簇
            labels[i] = clusterId;
            const queue = [i];

            this.steps.push({
                type: 'new_cluster',
                description: `开始新簇 ${clusterId}，从核心点 ${i} 开始`,
                points: points.map((p, idx) => ({
                    ...p,
                    label: labels[idx],
                    isCore: corePoints.has(idx),
                    current: idx === i
                })),
                currentCluster: clusterId
            });

            while (queue.length > 0) {
                const current = queue.shift();
                
                for (const neighbor of neighbors[current]) {
                    if (labels[neighbor] === -2) {
                        labels[neighbor] = clusterId;
                        
                        this.steps.push({
                            type: 'expand',
                            description: `将点 ${neighbor} 加入簇 ${clusterId}`,
                            points: points.map((p, idx) => ({
                                ...p,
                                label: labels[idx],
                                isCore: corePoints.has(idx),
                                current: idx === neighbor
                            })),
                            currentCluster: clusterId
                        });

                        if (corePoints.has(neighbor)) {
                            queue.push(neighbor);
                        }
                    }
                }
            }

            clusterId++;
        }

        // 标记噪声点
        for (let i = 0; i < n; i++) {
            if (labels[i] === -2) {
                labels[i] = -1;
            }
        }

        this.steps.push({
            type: 'complete',
            description: `聚类完成！共找到 ${clusterId} 个簇`,
            points: points.map((p, i) => ({
                ...p,
                label: labels[i],
                isCore: corePoints.has(i)
            })),
            numClusters: clusterId
        });

        return {labels, numClusters: clusterId, steps: this.steps};
    }

    distance(p1, p2) {
        const dx = p1.x - p2.x;
        const dy = p1.y - p2.y;
        return Math.sqrt(dx * dx + dy * dy);
    }
}

// ============================================================================
// 网格聚类算法
// ============================================================================
class GridClustering {
    constructor(gridSize, densityThreshold) {
        this.gridSize = gridSize;
        this.densityThreshold = densityThreshold;
        this.steps = [];
    }

    fit(points) {
        this.steps = [];
        const n = points.length;

        // 步骤1: 计算边界
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;
        
        for (const p of points) {
            minX = Math.min(minX, p.x);
            maxX = Math.max(maxX, p.x);
            minY = Math.min(minY, p.y);
            maxY = Math.max(maxY, p.y);
        }

        const cellWidth = (maxX - minX) / this.gridSize;
        const cellHeight = (maxY - minY) / this.gridSize;

        this.steps.push({
            type: 'init',
            description: '初始化网格',
            points: points.map(p => ({...p, label: -1})),
            grid: {minX, maxX, minY, maxY, cellWidth, cellHeight, size: this.gridSize}
        });

        // 步骤2: 将点分配到网格
        const grid = new Map();
        const pointToGrid = new Array(n);

        for (let i = 0; i < n; i++) {
            const p = points[i];
            const gridX = Math.min(Math.floor((p.x - minX) / cellWidth), this.gridSize - 1);
            const gridY = Math.min(Math.floor((p.y - minY) / cellHeight), this.gridSize - 1);
            const key = `${gridX},${gridY}`;
            
            if (!grid.has(key)) {
                grid.set(key, []);
            }
            grid.get(key).push(i);
            pointToGrid[i] = {x: gridX, y: gridY};
        }

        this.steps.push({
            type: 'assign_grid',
            description: `将 ${n} 个点分配到网格中，共 ${grid.size} 个非空网格`,
            points: points.map((p, i) => ({...p, label: -1, grid: pointToGrid[i]})),
            grid: {minX, maxX, minY, maxY, cellWidth, cellHeight, size: this.gridSize},
            gridCells: grid
        });

        // 步骤3: 识别密集网格
        const denseGrids = new Set();
        for (const [key, pointIndices] of grid.entries()) {
            if (pointIndices.length >= this.densityThreshold) {
                denseGrids.add(key);
            }
        }

        this.steps.push({
            type: 'find_dense',
            description: `找到 ${denseGrids.size} 个密集网格（阈值=${this.densityThreshold}）`,
            points: points.map((p, i) => ({...p, label: -1, grid: pointToGrid[i]})),
            grid: {minX, maxX, minY, maxY, cellWidth, cellHeight, size: this.gridSize},
            gridCells: grid,
            denseGrids: denseGrids
        });

        // 步骤4: 连接相邻的密集网格形成簇
        const labels = new Array(n).fill(-1);
        const visited = new Set();
        let clusterId = 0;

        const getNeighbors = (key) => {
            const [x, y] = key.split(',').map(Number);
            const neighbors = [];
            for (let dx = -1; dx <= 1; dx++) {
                for (let dy = -1; dy <= 1; dy++) {
                    if (dx === 0 && dy === 0) continue;
                    const nx = x + dx;
                    const ny = y + dy;
                    if (nx >= 0 && nx < this.gridSize && ny >= 0 && ny < this.gridSize) {
                        neighbors.push(`${nx},${ny}`);
                    }
                }
            }
            return neighbors;
        };

        for (const key of denseGrids) {
            if (visited.has(key)) continue;

            const queue = [key];
            visited.add(key);

            this.steps.push({
                type: 'new_cluster',
                description: `开始新簇 ${clusterId}，从网格 ${key} 开始`,
                points: points.map((p, i) => ({...p, label: labels[i], grid: pointToGrid[i]})),
                grid: {minX, maxX, minY, maxY, cellWidth, cellHeight, size: this.gridSize},
                gridCells: grid,
                denseGrids: denseGrids,
                currentCluster: clusterId,
                currentGrid: key
            });

            while (queue.length > 0) {
                const current = queue.shift();
                
                // 标记该网格中的所有点
                if (grid.has(current)) {
                    for (const idx of grid.get(current)) {
                        labels[idx] = clusterId;
                    }
                }

                // 检查相邻网格
                for (const neighbor of getNeighbors(current)) {
                    if (denseGrids.has(neighbor) && !visited.has(neighbor)) {
                        visited.add(neighbor);
                        queue.push(neighbor);
                    }
                }
            }

            this.steps.push({
                type: 'cluster_complete',
                description: `簇 ${clusterId} 完成`,
                points: points.map((p, i) => ({...p, label: labels[i], grid: pointToGrid[i]})),
                grid: {minX, maxX, minY, maxY, cellWidth, cellHeight, size: this.gridSize},
                gridCells: grid,
                denseGrids: denseGrids,
                currentCluster: clusterId
            });

            clusterId++;
        }

        this.steps.push({
            type: 'complete',
            description: `聚类完成！共找到 ${clusterId} 个簇`,
            points: points.map((p, i) => ({...p, label: labels[i], grid: pointToGrid[i]})),
            grid: {minX, maxX, minY, maxY, cellWidth, cellHeight, size: this.gridSize},
            gridCells: grid,
            denseGrids: denseGrids,
            numClusters: clusterId
        });

        return {labels, numClusters: clusterId, steps: this.steps};
    }
}

// ============================================================================
// GMM 高斯混合模型（简化版）
// ============================================================================
class GMM {
    constructor(k, maxIter = 20) {
        this.k = k;
        this.maxIter = maxIter;
        this.steps = [];
    }

    fit(points) {
        this.steps = [];
        const n = points.length;

        // 步骤1: 初始化 - K-means++方式选择初始中心
        const means = this.initializeMeans(points);
        const covariances = Array(this.k).fill(null).map(() => [[1, 0], [0, 1]]);
        const weights = Array(this.k).fill(1 / this.k);

        this.steps.push({
            type: 'init',
            description: `初始化 ${this.k} 个高斯分量`,
            points: points.map(p => ({...p, label: -1, probs: []})),
            means: means,
            covariances: covariances,
            weights: weights
        });

        let labels = new Array(n).fill(0);

        // EM算法迭代
        for (let iter = 0; iter < this.maxIter; iter++) {
            // E步: 计算每个点属于各个簇的概率
            const responsibilities = [];
            
            for (let i = 0; i < n; i++) {
                const p = points[i];
                const probs = [];
                let sum = 0;

                for (let j = 0; j < this.k; j++) {
                    const prob = weights[j] * this.gaussianPDF(p, means[j], covariances[j]);
                    probs.push(prob);
                    sum += prob;
                }

                // 归一化
                const normalized = probs.map(pr => pr / (sum + 1e-10));
                responsibilities.push(normalized);
                labels[i] = this.argmax(normalized);
            }

            this.steps.push({
                type: 'e_step',
                description: `迭代 ${iter + 1}: E步 - 计算每个点的归属概率`,
                points: points.map((p, i) => ({
                    ...p,
                    label: labels[i],
                    probs: responsibilities[i]
                })),
                means: means.map(m => ({...m})),
                covariances: covariances.map(c => c.map(row => [...row])),
                weights: [...weights],
                iteration: iter + 1
            });

            // M步: 更新参数
            for (let j = 0; j < this.k; j++) {
                let nj = 0;
                let newMean = {x: 0, y: 0};

                for (let i = 0; i < n; i++) {
                    const r = responsibilities[i][j];
                    nj += r;
                    newMean.x += r * points[i].x;
                    newMean.y += r * points[i].y;
                }

                if (nj > 0) {
                    newMean.x /= nj;
                    newMean.y /= nj;
                    means[j] = newMean;

                    // 更新协方差
                    let cov = [[0, 0], [0, 0]];
                    for (let i = 0; i < n; i++) {
                        const r = responsibilities[i][j];
                        const dx = points[i].x - newMean.x;
                        const dy = points[i].y - newMean.y;
                        cov[0][0] += r * dx * dx;
                        cov[0][1] += r * dx * dy;
                        cov[1][0] += r * dy * dx;
                        cov[1][1] += r * dy * dy;
                    }
                    cov = cov.map(row => row.map(val => val / nj + 0.01)); // 添加正则化
                    covariances[j] = cov;

                    weights[j] = nj / n;
                }
            }

            this.steps.push({
                type: 'm_step',
                description: `迭代 ${iter + 1}: M步 - 更新高斯分量参数`,
                points: points.map((p, i) => ({
                    ...p,
                    label: labels[i],
                    probs: responsibilities[i]
                })),
                means: means.map(m => ({...m})),
                covariances: covariances.map(c => c.map(row => [...row])),
                weights: [...weights],
                iteration: iter + 1
            });
        }

        this.steps.push({
            type: 'complete',
            description: `EM算法收敛！共 ${this.k} 个簇`,
            points: points.map((p, i) => ({...p, label: labels[i]})),
            means: means,
            covariances: covariances,
            weights: weights,
            numClusters: this.k
        });

        return {labels, numClusters: this.k, means, covariances, weights, steps: this.steps};
    }

    initializeMeans(points) {
        const means = [];
        const n = points.length;
        
        // 随机选择第一个中心
        means.push({...points[Math.floor(Math.random() * n)]});

        // K-means++: 选择距离已有中心最远的点
        for (let i = 1; i < this.k; i++) {
            let maxDist = -1;
            let farthest = null;

            for (const p of points) {
                let minDist = Infinity;
                for (const mean of means) {
                    const dist = this.distance(p, mean);
                    minDist = Math.min(minDist, dist);
                }
                if (minDist > maxDist) {
                    maxDist = minDist;
                    farthest = p;
                }
            }

            means.push({...farthest});
        }

        return means;
    }

    gaussianPDF(point, mean, covariance) {
        const dx = point.x - mean.x;
        const dy = point.y - mean.y;
        
        // 计算行列式
        const det = covariance[0][0] * covariance[1][1] - covariance[0][1] * covariance[1][0];
        if (det <= 0) return 1e-10;

        // 计算逆矩阵
        const invCov = [
            [covariance[1][1] / det, -covariance[0][1] / det],
            [-covariance[1][0] / det, covariance[0][0] / det]
        ];

        // 计算马氏距离
        const mahal = dx * (invCov[0][0] * dx + invCov[0][1] * dy) + 
                      dy * (invCov[1][0] * dx + invCov[1][1] * dy);

        return Math.exp(-0.5 * mahal) / (2 * Math.PI * Math.sqrt(det));
    }

    distance(p1, p2) {
        const dx = p1.x - p2.x;
        const dy = p1.y - p2.y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    argmax(arr) {
        let maxIdx = 0;
        let maxVal = arr[0];
        for (let i = 1; i < arr.length; i++) {
            if (arr[i] > maxVal) {
                maxVal = arr[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }
}

