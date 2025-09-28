"""
Google Cloud Storage client for photo access and management.

Provides async interface for downloading and processing photos from GCS.
"""

import io
import aiohttp
import asyncio
from typing import Optional, Tuple
from urllib.parse import urlparse

import structlog
from google.cloud import storage
from google.cloud.exceptions import NotFound, Forbidden
from PIL import Image

from fn_media_ai.infrastructure.config.settings import Settings


class GCSClient:
    """
    Google Cloud Storage client for photo operations.

    Provides async interface for accessing photos stored in GCS
    with proper error handling and image validation.
    """

    def __init__(self, settings: Settings):
        """
        Initialize GCS client.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.bucket_name = settings.gcs_bucket_name
        self.project_id = settings.gcs_project_id
        self.logger = structlog.get_logger()

        # Initialize GCS client
        self._client = storage.Client(project=self.project_id)
        self._bucket = self._client.bucket(self.bucket_name)

    async def download_photo(self, photo_url: str) -> Optional[bytes]:
        """
        Download photo from GCS or HTTP URL.

        Args:
            photo_url: Photo URL to download

        Returns:
            bytes: Photo data or None if download fails

        Raises:
            PhotoDownloadError: When download fails
        """
        logger = self.logger.bind(photo_url=photo_url)

        try:
            # Parse URL to determine download method
            parsed_url = urlparse(photo_url)

            if 'storage.googleapis.com' in parsed_url.netloc or 'storage.cloud.google.com' in parsed_url.netloc:
                # GCS URL - use GCS client
                return await self._download_from_gcs(photo_url, logger)
            else:
                # HTTP URL - use aiohttp
                return await self._download_from_http(photo_url, logger)

        except Exception as e:
            logger.error("Photo download failed", error=str(e))
            raise PhotoDownloadError(f"Failed to download photo: {str(e)}")

    async def _download_from_gcs(self, photo_url: str, logger) -> bytes:
        """Download photo from GCS bucket."""
        # Extract blob name from GCS URL
        blob_name = self._extract_blob_name_from_url(photo_url)

        logger = logger.bind(bucket=self.bucket_name, blob_name=blob_name)
        logger.debug("Downloading from GCS")

        try:
            # Run GCS download in thread pool to avoid blocking
            blob_data = await asyncio.get_event_loop().run_in_executor(
                None, self._download_blob_sync, blob_name
            )

            logger.debug("GCS download completed", size_bytes=len(blob_data))
            return blob_data

        except NotFound:
            logger.warning("Photo not found in GCS")
            raise PhotoDownloadError("Photo not found in GCS")
        except Forbidden:
            logger.warning("Access forbidden to GCS photo")
            raise PhotoDownloadError("Access forbidden to GCS photo")

    def _download_blob_sync(self, blob_name: str) -> bytes:
        """Synchronous blob download for thread pool execution."""
        blob = self._bucket.blob(blob_name)
        return blob.download_as_bytes()

    async def _download_from_http(self, photo_url: str, logger) -> bytes:
        """Download photo from HTTP URL."""
        logger.debug("Downloading from HTTP")

        timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(photo_url) as response:
                    if response.status == 200:
                        photo_data = await response.read()
                        logger.debug("HTTP download completed", size_bytes=len(photo_data))
                        return photo_data
                    else:
                        logger.warning("HTTP download failed", status_code=response.status)
                        raise PhotoDownloadError(f"HTTP download failed with status {response.status}")

        except aiohttp.ClientError as e:
            logger.warning("HTTP client error", error=str(e))
            raise PhotoDownloadError(f"HTTP download failed: {str(e)}")
        except asyncio.TimeoutError:
            logger.warning("HTTP download timeout")
            raise PhotoDownloadError("HTTP download timeout")

    def _extract_blob_name_from_url(self, gcs_url: str) -> str:
        """Extract blob name from GCS URL."""
        parsed = urlparse(gcs_url)

        # Handle different GCS URL formats
        if 'storage.googleapis.com' in parsed.netloc:
            # Format: https://storage.googleapis.com/bucket-name/path/to/object
            path_parts = parsed.path.strip('/').split('/', 1)
            if len(path_parts) > 1:
                return path_parts[1]
        elif 'storage.cloud.google.com' in parsed.netloc:
            # Format: https://storage.cloud.google.com/bucket-name/path/to/object
            return parsed.path.strip('/')

        # Fallback - assume path is the blob name
        return parsed.path.strip('/')

    async def validate_photo(self, photo_url: str) -> Tuple[bool, Optional[dict]]:
        """
        Validate photo accessibility and quality.

        Args:
            photo_url: Photo URL to validate

        Returns:
            Tuple[bool, Optional[dict]]: (is_valid, metadata)
        """
        logger = self.logger.bind(photo_url=photo_url)

        try:
            # Download photo data
            photo_data = await self.download_photo(photo_url)

            if not photo_data:
                return False, None

            # Validate image format and get metadata
            metadata = await self._validate_image_data(photo_data, logger)

            if metadata:
                logger.debug("Photo validation successful", metadata=metadata)
                return True, metadata
            else:
                logger.warning("Photo validation failed")
                return False, None

        except Exception as e:
            logger.warning("Photo validation error", error=str(e))
            return False, None

    async def _validate_image_data(self, image_data: bytes, logger) -> Optional[dict]:
        """Validate image data and extract metadata."""
        try:
            # Run image processing in thread pool
            metadata = await asyncio.get_event_loop().run_in_executor(
                None, self._process_image_sync, image_data
            )

            return metadata

        except Exception as e:
            logger.warning("Image processing failed", error=str(e))
            return None

    def _process_image_sync(self, image_data: bytes) -> dict:
        """Synchronous image processing for thread pool execution."""
        # Open image with PIL
        image = Image.open(io.BytesIO(image_data))

        # Validate image
        if image.format not in ['JPEG', 'PNG', 'WEBP', 'BMP']:
            raise ValueError(f"Unsupported image format: {image.format}")

        # Check image size constraints
        max_size_mb = self.settings.max_photo_size_mb
        size_mb = len(image_data) / (1024 * 1024)

        if size_mb > max_size_mb:
            raise ValueError(f"Image too large: {size_mb:.1f}MB > {max_size_mb}MB")

        # Check minimum dimensions
        width, height = image.size
        if width < 100 or height < 100:
            raise ValueError(f"Image too small: {width}x{height} < 100x100")

        # Extract metadata
        metadata = {
            'format': image.format,
            'mode': image.mode,
            'width': width,
            'height': height,
            'size_bytes': len(image_data),
            'size_mb': round(size_mb, 2),
        }

        # Try to extract EXIF data
        if hasattr(image, '_getexif') and image._getexif():
            exif_data = image._getexif()
            metadata['has_exif'] = True

            # Extract GPS coordinates if available
            gps_info = exif_data.get(34853)  # GPS Info tag
            if gps_info:
                try:
                    lat, lon = self._extract_gps_coordinates(gps_info)
                    if lat and lon:
                        metadata['gps_latitude'] = lat
                        metadata['gps_longitude'] = lon
                except Exception:
                    # GPS extraction failed, continue without GPS
                    pass

        return metadata

    def _extract_gps_coordinates(self, gps_info: dict) -> Tuple[Optional[float], Optional[float]]:
        """Extract GPS coordinates from EXIF data."""
        def convert_to_degrees(value):
            """Convert GPS coordinates to decimal degrees."""
            d = float(value[0])
            m = float(value[1])
            s = float(value[2])
            return d + (m / 60.0) + (s / 3600.0)

        try:
            lat = lon = None

            if 2 in gps_info and 1 in gps_info:  # Latitude
                lat = convert_to_degrees(gps_info[2])
                if gps_info[1] == 'S':
                    lat = -lat

            if 4 in gps_info and 3 in gps_info:  # Longitude
                lon = convert_to_degrees(gps_info[4])
                if gps_info[3] == 'W':
                    lon = -lon

            return lat, lon

        except Exception:
            return None, None

    async def get_photo_metadata(self, photo_url: str) -> Optional[dict]:
        """
        Get photo metadata without full download.

        Args:
            photo_url: Photo URL

        Returns:
            dict: Photo metadata or None
        """
        try:
            # For GCS URLs, we can get metadata without downloading
            if 'storage.googleapis.com' in photo_url or 'storage.cloud.google.com' in photo_url:
                return await self._get_gcs_metadata(photo_url)
            else:
                # For HTTP URLs, we need to download to get metadata
                return await self._get_http_metadata(photo_url)

        except Exception as e:
            self.logger.warning("Failed to get photo metadata", photo_url=photo_url, error=str(e))
            return None

    async def _get_gcs_metadata(self, photo_url: str) -> dict:
        """Get metadata for GCS object."""
        blob_name = self._extract_blob_name_from_url(photo_url)

        # Run in thread pool
        metadata = await asyncio.get_event_loop().run_in_executor(
            None, self._get_blob_metadata_sync, blob_name
        )

        return metadata

    def _get_blob_metadata_sync(self, blob_name: str) -> dict:
        """Get GCS blob metadata synchronously."""
        blob = self._bucket.blob(blob_name)
        blob.reload()

        return {
            'name': blob.name,
            'size_bytes': blob.size,
            'content_type': blob.content_type,
            'created': blob.time_created.isoformat() if blob.time_created else None,
            'updated': blob.updated.isoformat() if blob.updated else None,
        }

    async def _get_http_metadata(self, photo_url: str) -> dict:
        """Get HTTP photo metadata via HEAD request."""
        timeout = aiohttp.ClientTimeout(total=10)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.head(photo_url) as response:
                if response.status == 200:
                    return {
                        'content_type': response.headers.get('Content-Type'),
                        'content_length': int(response.headers.get('Content-Length', 0)),
                        'last_modified': response.headers.get('Last-Modified'),
                    }
                else:
                    raise PhotoDownloadError(f"HTTP HEAD request failed with status {response.status}")


class PhotoDownloadError(Exception):
    """Raised when photo download fails."""
    pass