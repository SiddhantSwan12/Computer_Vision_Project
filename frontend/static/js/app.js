/**
 * Manufacturing Vision System - JavaScript Functions
 * Handles interactive features for the web interface
 */

document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const videoFeed = document.getElementById('videoFeed');
    const videoOverlay = document.getElementById('videoOverlay');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const toastContainer = document.getElementById('toastContainer');
    const sidebar = document.getElementById('sidebar');
    
    // Video source controls
    const sampleVideoBtn = document.getElementById('sampleVideoBtn');
    const webcamBtn = document.getElementById('webcamBtn');
    const customVideoBtn = document.getElementById('customVideoBtn');
    const customVideoFile = document.getElementById('customVideoFile');
    const sourceLabel = document.getElementById('sourceLabel');
    
    // Display options
    const showBoxes = document.getElementById('showBoxes');
    const showContours = document.getElementById('showContours');
    const showIDs = document.getElementById('showIDs');
    
    // Statistics elements
    const fpsValue = document.getElementById('fpsValue');
    const objectsValue = document.getElementById('objectsValue');
    const processingTimeValue = document.getElementById('processingTimeValue');
    const statusValue = document.getElementById('statusValue');
    const objectList = document.getElementById('objectList');
    const objectCount = document.getElementById('objectCount');
    
    // Video controls
    const screenshotBtn = document.getElementById('screenshotBtn');
    const fullscreenBtn = document.getElementById('fullscreenBtn');
    const clearObjectsBtn = document.getElementById('clearObjectsBtn');
    
    // Mobile menu controls
    const mobileMenuToggle = document.getElementById('mobileMenuToggle');
    const closeSidebar = document.getElementById('closeSidebar');
    
    // Dark mode toggle
    const darkModeToggle = document.getElementById('darkModeToggle');
    
    // Screenshot modal elements
    const screenshotModal = document.getElementById('screenshotModal');
    const closeModal = document.getElementById('closeModal');
    const screenshotPreview = document.getElementById('screenshotPreview');
    const downloadScreenshot = document.getElementById('downloadScreenshot');
    const cancelScreenshot = document.getElementById('cancelScreenshot');
    
    // Anomaly detection elements
    const anomalyToggle = document.getElementById('anomalyToggle');
    const referenceImageInput = document.getElementById('referenceImageInput');
    const uploadReferenceImageBtn = document.getElementById('uploadReferenceImageBtn');
    
    // Set current year in footer
    document.getElementById('currentYear').textContent = new Date().getFullYear();
    
    // Initialize the application
    initApp();
    
    function initApp() {
        // Check video feed status
        checkVideoFeedStatus();
        
        // Check for available cameras
        checkAvailableCameras();
        
        // Set up event listeners
        setupEventListeners();
        
        // Initialize dark mode
        initDarkMode();
        
        // Start stats update interval
        setInterval(updateStats, 1000);
        
        // Initial stats update
        updateStats();
        
        // Show video feed
        showVideoFeed();
    }
    
    function setupEventListeners() {
        // Video source selection
        sampleVideoBtn.addEventListener('click', () => changeVideoSource('sample'));
        webcamBtn.addEventListener('click', () => changeVideoSource('webcam'));
        customVideoBtn.addEventListener('click', handleCustomVideoUpload);
        
        // Display options
        showBoxes.addEventListener('change', updateDisplaySettings);
        showContours.addEventListener('change', updateDisplaySettings);
        showIDs.addEventListener('change', updateDisplaySettings);
        
        // Video controls
        screenshotBtn.addEventListener('click', takeScreenshot);
        fullscreenBtn.addEventListener('click', toggleFullscreen);
        clearObjectsBtn.addEventListener('click', clearObjects);
        
        // Mobile menu
        mobileMenuToggle.addEventListener('click', toggleSidebar);
        closeSidebar.addEventListener('click', toggleSidebar);
        
        // Dark mode toggle
        darkModeToggle.addEventListener('click', toggleDarkMode);
        
        // Screenshot modal
        closeModal.addEventListener('click', closeScreenshotModal);
        downloadScreenshot.addEventListener('click', downloadScreenshotImage);
        cancelScreenshot.addEventListener('click', closeScreenshotModal);
        
        // Video loaded event
        videoFeed.addEventListener('load', hideLoading);
        
        // Document click handler for closing sidebar on mobile
        document.addEventListener('click', function(event) {
            if (window.innerWidth <= 768 && 
                sidebar.classList.contains('active') && 
                !sidebar.contains(event.target) && 
                !mobileMenuToggle.contains(event.target)) {
                sidebar.classList.remove('active');
            }
        });
        
        // Handle escape key for modals
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape') {
                closeScreenshotModal();
                
                if (window.innerWidth <= 768 && sidebar.classList.contains('active')) {
                    sidebar.classList.remove('active');
                }
            }
        });
        
        // Handle fullscreen change
        document.addEventListener('fullscreenchange', updateFullscreenButton);
        
        // Upload reference image
        uploadReferenceImageBtn.addEventListener('click', () => {
            const file = referenceImageInput.files[0];
            if (!file) {
                alert('Please select a reference image.');
                return;
            }
    
            const formData = new FormData();
            formData.append('image', file);
    
            fetch('/upload_reference_image', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('Reference image uploaded successfully.');
                } else {
                    alert(`Error: ${data.error}`);
                }
            })
            .catch(error => {
                console.error('Error uploading reference image:', error);
            });
        });
    
        // Toggle anomaly detection
        anomalyToggle.addEventListener('change', () => {
            const enabled = anomalyToggle.checked;
    
            fetch('/toggle_anomaly_detection', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ enabled })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    console.log('Anomaly detection toggled:', enabled);
                } else {
                    console.error('Error toggling anomaly detection:', data.error);
                }
            })
            .catch(error => {
                console.error('Error toggling anomaly detection:', error);
            });
        });
    }
    
    function initDarkMode() {
        // Check for saved preference
        const darkModeEnabled = localStorage.getItem('darkMode') === 'true';
        
        if (darkModeEnabled) {
            document.body.classList.add('dark-mode');
            darkModeToggle.innerHTML = '<i class="fas fa-sun"></i>';
        } else {
            document.body.classList.remove('dark-mode');
            darkModeToggle.innerHTML = '<i class="fas fa-moon"></i>';
        }
    }
    
    function toggleDarkMode() {
        const isDarkMode = document.body.classList.toggle('dark-mode');
        
        // Add transition class
        document.body.classList.add('dark-mode-transition');
        
        // Update button icon
        if (isDarkMode) {
            darkModeToggle.innerHTML = '<i class="fas fa-sun"></i>';
        } else {
            darkModeToggle.innerHTML = '<i class="fas fa-moon"></i>';
        }
        
        // Save preference
        localStorage.setItem('darkMode', isDarkMode);
        
        // Remove transition class after animation completes
        setTimeout(() => {
            document.body.classList.remove('dark-mode-transition');
        }, 500);
        
        // Show toast notification
        showToast(
            isDarkMode ? 'Dark Mode Enabled' : 'Light Mode Enabled', 
            'Theme preference saved', 
            isDarkMode ? 'success' : 'info'
        );
    }
    
    function toggleSidebar() {
        sidebar.classList.toggle('active');
    }
    
    function checkAvailableCameras() {
        fetch('/check_cameras')
            .then(response => response.json())
            .then(data => {
                if (data.success && data.count > 0) {
                    console.log(`Found ${data.count} camera(s): ${data.available_cameras.join(', ')}`);
                    webcamBtn.disabled = false;
                    webcamBtn.title = `${data.count} camera(s) available`;
                } else {
                    console.warn('No cameras found or error checking cameras');
                    webcamBtn.disabled = true;
                    webcamBtn.title = 'No cameras available';
                    webcamBtn.classList.add('disabled');
                }
            })
            .catch(error => {
                console.error('Error checking cameras:', error);
                webcamBtn.disabled = true;
                webcamBtn.title = 'Error checking cameras';
                webcamBtn.classList.add('disabled');
                showToast('Camera Error', 'Could not detect available cameras', 'error');
            });
    }
    
    function changeVideoSource(source) {
        // Update active button state
        sampleVideoBtn.classList.remove('active');
        webcamBtn.classList.remove('active');
        customVideoBtn.classList.remove('active');
        sampleVideoBtn.setAttribute('aria-pressed', 'false');
        webcamBtn.setAttribute('aria-pressed', 'false');
        customVideoBtn.setAttribute('aria-pressed', 'false');
        
        showLoading();
        
        // Add timestamp to prevent caching
        const timestamp = new Date().getTime();
        let retryCount = 0;
        const maxRetries = 3;
        
        function tryConnect() {
            if (source === 'sample') {
                // Use the default video feed endpoint
                videoFeed.src = `/video_feed?timestamp=${timestamp}`;
                sourceLabel.textContent = 'Sample Video';
                sampleVideoBtn.classList.add('active');
                sampleVideoBtn.setAttribute('aria-pressed', 'true');
                
                // Show toast notification
                showToast('Video Source Changed', 'Switched to sample video', 'info');
            } else if (source === 'webcam') {
                // Use the webcam feed endpoint
                videoFeed.src = `/video_feed?source=webcam&timestamp=${timestamp}`;
                sourceLabel.textContent = 'Webcam';
                webcamBtn.classList.add('active');
                webcamBtn.setAttribute('aria-pressed', 'true');
                
                // Show toast notification
                showToast('Video Source Changed', 'Attempting to connect to webcam...', 'info');
            }
        }
        
        // Add error handling for video feed
        videoFeed.onerror = function() {
            if (source === 'webcam' && retryCount < maxRetries) {
                retryCount++;
                showToast('Webcam Error', `Retrying connection (${retryCount}/${maxRetries})...`, 'warning');
                setTimeout(tryConnect, 1000);  // Wait 1 second before retrying
            } else {
                showToast('Video Error', 'Failed to load video feed. Please check your camera connection and permissions.', 'error');
                hideLoading();
                
                // Reset active state if webcam fails
                if (source === 'webcam') {
                    webcamBtn.classList.remove('active');
                    webcamBtn.setAttribute('aria-pressed', 'false');
                    sourceLabel.textContent = 'No Source';
                }
            }
        };
        
        // Add load event handler
        videoFeed.onload = function() {
            hideLoading();
            if (source === 'webcam') {
                showToast('Webcam Connected', 'Successfully connected to webcam', 'success');
            }
        };
        
        // Initial connection attempt
        tryConnect();
    }
    
    function handleCustomVideoUpload() {
        if (customVideoFile.files.length > 0) {
            const formData = new FormData();
            const fileName = customVideoFile.files[0].name;
            formData.append('video', customVideoFile.files[0]);
            
            showLoading();
            sourceLabel.textContent = `Custom Video: ${fileName}`;
            sampleVideoBtn.classList.remove('active');
            webcamBtn.classList.remove('active');
            
            fetch('/upload_video', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    videoFeed.src = `/video_feed?source=custom&timestamp=${new Date().getTime()}`;
                    showToast('Video Uploaded', `Successfully loaded ${fileName}`, 'success');
                } else {
                    showToast('Upload Error', data.error || 'Error uploading video', 'error');
                    hideLoading();
                }
            })
            .catch(error => {
                console.error('Error:', error);
                showToast('Upload Error', 'Failed to upload video', 'error');
                hideLoading();
            });
        } else {
            showToast('No File Selected', 'Please select a video file first', 'warning');
        }
    }
    
    function updateDisplaySettings() {
        const settings = {
            show_boxes: showBoxes.checked,
            show_contours: showContours.checked,
            show_ids: showIDs.checked
        };
        
        fetch('/update_display?' + new URLSearchParams(settings).toString())
            .then(response => response.json())
            .then(data => {
                console.log('Display settings updated:', data);
            })
            .catch(error => {
                console.error('Error updating display settings:', error);
                showToast('Settings Error', 'Failed to update display settings', 'error');
            });
    }
    
    function updateStats() {
        fetch('/get_stats')
            .then(response => response.json())
            .then(data => {
                // Update statistics
                fpsValue.textContent = data.fps;
                objectsValue.textContent = data.objects_detected;
                processingTimeValue.textContent = data.processing_time_ms + ' ms';
                
                // Update object count
                objectCount.textContent = `${data.objects ? data.objects.length : 0} objects`;
                
                // Update object list
                if (data.objects && data.objects.length > 0) {
                    let objectListHTML = '';
                    for (const obj of data.objects) {
                        objectListHTML += `
                            <div class="object-item">
                                <div class="object-id">ID: ${obj.id}</div>
                                <div class="object-info">
                                    <div class="object-info-item">
                                        <span class="object-info-label">Position</span>
                                        <span>(${obj.position.x}, ${obj.position.y})</span>
                                    </div>
                                    <div class="object-info-item">
                                        <span class="object-info-label">Size</span>
                                        <span>${obj.size.width}px × ${obj.size.height}px</span>
                                    </div>
                                    <div class="object-info-item">
                                        <span class="object-info-label">Confidence</span>
                                        <span>${(obj.confidence * 100).toFixed(1)}%</span>
                                    </div>
                                    <div class="object-info-item">
                                        <span class="object-info-label">Class</span>
                                        <span>${obj.class || 'Unknown'}</span>
                                    </div>
                                </div>
                            </div>
                        `;
                    }
                    objectList.innerHTML = objectListHTML;
                } else {
                    objectList.innerHTML = '<p class="no-objects">No objects detected</p>';
                }
            })
            .catch(error => {
                console.error('Error fetching stats:', error);
                statusValue.textContent = 'Error';
                statusValue.classList.remove('status-active');
                statusValue.style.color = 'var(--color-danger)';
            });
    }
    
    function takeScreenshot() {
        const canvas = document.createElement('canvas');
        canvas.width = videoFeed.naturalWidth || videoFeed.clientWidth;
        canvas.height = videoFeed.naturalHeight || videoFeed.clientHeight;
        
        const ctx = canvas.getContext('2d');
        ctx.drawImage(videoFeed, 0, 0, canvas.width, canvas.height);
        
        // Show screenshot in modal
        screenshotPreview.src = canvas.toDataURL('image/png');
        screenshotModal.style.display = 'flex';
    }
    
    function downloadScreenshotImage() {
        const link = document.createElement('a');
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        link.download = `manufacturing-vision-${timestamp}.png`;
        link.href = screenshotPreview.src;
        link.click();
        
        closeScreenshotModal();
        showToast('Screenshot Saved', 'Image has been downloaded', 'success');
    }
    
    function closeScreenshotModal() {
        screenshotModal.style.display = 'none';
    }
    
    function toggleFullscreen() {
        const videoContainer = document.getElementById('videoContainer');
        
        if (!document.fullscreenElement) {
            if (videoContainer.requestFullscreen) {
                videoContainer.requestFullscreen();
            } else if (videoContainer.webkitRequestFullscreen) {
                videoContainer.webkitRequestFullscreen();
            } else if (videoContainer.msRequestFullscreen) {
                videoContainer.msRequestFullscreen();
            }
        } else {
            if (document.exitFullscreen) {
                document.exitFullscreen();
            } else if (document.webkitExitFullscreen) {
                document.webkitExitFullscreen();
            } else if (document.msExitFullscreen) {
                document.msExitFullscreen();
            }
        }
    }
    
    function updateFullscreenButton() {
        if (document.fullscreenElement) {
            fullscreenBtn.innerHTML = '<i class="fas fa-compress"></i>';
            fullscreenBtn.setAttribute('data-tooltip', 'Exit Fullscreen');
        } else {
            fullscreenBtn.innerHTML = '<i class="fas fa-expand"></i>';
            fullscreenBtn.setAttribute('data-tooltip', 'Fullscreen');
        }
    }
    
    function clearObjects() {
        fetch('/clear_objects')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    objectList.innerHTML = '<p class="no-objects">No objects detected</p>';
                    objectCount.textContent = '0 objects';
                    showToast('Objects Cleared', 'All tracked objects have been cleared', 'info');
                }
            })
            .catch(error => {
                console.error('Error clearing objects:', error);
                showToast('Error', 'Failed to clear objects', 'error');
            });
    }
    
    function showVideoFeed() {
        videoFeed.style.opacity = '1';
        videoOverlay.style.display = 'none';
    }
    
    function showLoading() {
        videoOverlay.style.display = 'flex';
        loadingOverlay.style.display = 'flex';
    }
    
    function hideLoading() {
        videoOverlay.style.display = 'none';
        loadingOverlay.style.display = 'none';
    }
    
    function showToast(title, message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <div class="toast-content">
                <div class="toast-title">${title}</div>
                <div class="toast-message">${message}</div>
            </div>
            <button class="toast-close">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        // Add to container
        toastContainer.appendChild(toast);
        
        // Add close event
        const closeBtn = toast.querySelector('.toast-close');
        closeBtn.addEventListener('click', () => {
            toast.style.animation = 'slideOut 0.3s ease forwards';
            setTimeout(() => {
                toastContainer.removeChild(toast);
            }, 300);
        });
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (toast.parentNode === toastContainer) {
                toast.style.animation = 'slideOut 0.3s ease forwards';
                setTimeout(() => {
                    if (toast.parentNode === toastContainer) {
                        toastContainer.removeChild(toast);
                    }
                }, 300);
            }
        }, 5000);
    }
    
    function checkVideoFeedStatus() {
        fetch('/check_video_feed')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    if (!data.video_exists) {
                        showToast('Video Warning', 'Sample video not found, but you can still use other sources.', 'warning');
                    }
                    
                    if (!data.webcam_available) {
                        showToast('Camera Error', 'No webcam detected', 'warning');
                        webcamBtn.disabled = true;
                        webcamBtn.title = 'No webcam available';
                        webcamBtn.classList.add('disabled');
                    }
                } else {
                    showToast('Error', data.error || 'Failed to check video feed status', 'error');
                }
            })
            .catch(error => {
                console.error('Error checking video feed:', error);
                showToast('Error', 'Failed to check video feed status', 'error');
            });
    }
});