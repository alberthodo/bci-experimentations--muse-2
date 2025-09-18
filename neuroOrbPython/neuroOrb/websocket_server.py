"""
WebSocket Server for BCI Orb Control Project
Handles real-time communication with Unity frontend
"""

import asyncio
import websockets
import json
import time
import logging
from typing import Dict, Any, Set
from datetime import datetime


class BCIWebSocketServer:
    """
    WebSocket server for real-time BCI data communication
    """
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        """
        Initialize WebSocket server
        
        Args:
            host: Server host address
            port: Server port number
        """
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.logger = logging.getLogger(__name__)
        
        # Message types for the protocol
        self.MESSAGE_TYPES = {
            "MENTAL_STATE": "mental_state",
            "DOUBLE_BLINK": "double_blink", 
            "SINGLE_BLINK": "single_blink",
            "SYSTEM_STATUS": "system_status",
            "ERROR": "error"
        }
        
        self.logger.info(f"WebSocket server initialized on {host}:{port}")
    
    async def register_client(self, websocket: websockets.WebSocketServerProtocol):
        """Register a new client connection"""
        self.clients.add(websocket)
        self.logger.info(f"Client connected. Total clients: {len(self.clients)}")
        
        # Send welcome message
        welcome_msg = self._create_message(
            self.MESSAGE_TYPES["SYSTEM_STATUS"],
            {"status": "connected", "message": "BCI WebSocket server connected"}
        )
        await self._send_to_client(websocket, welcome_msg)
    
    async def unregister_client(self, websocket: websockets.WebSocketServerProtocol):
        """Unregister a client connection"""
        self.clients.discard(websocket)
        self.logger.info(f"Client disconnected. Total clients: {len(self.clients)}")
    
    async def handle_client(self, websocket):
        """Handle individual client connection"""
        await self.register_client(websocket)
        
        try:
            async for message in websocket:
                await self._handle_incoming_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            self.logger.error(f"Error handling client: {e}")
        finally:
            await self.unregister_client(websocket)

    
    async def _handle_incoming_message(self, websocket: websockets.WebSocketServerProtocol, message: str):
        """Handle incoming messages from clients"""
        try:
            data = json.loads(message)
            self.logger.debug(f"Received message: {data}")
            
            # Handle different message types from Unity
            if data.get("type") == "ping":
                # Respond to ping with pong
                pong_msg = self._create_message(
                    self.MESSAGE_TYPES["SYSTEM_STATUS"],
                    {"status": "pong", "timestamp": time.time()}
                )
                await self._send_to_client(websocket, pong_msg)
                
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON received: {message}")
        except Exception as e:
            self.logger.error(f"Error handling incoming message: {e}")
    
    def _create_message(self, message_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a standardized message
        
        Args:
            message_type: Type of message (mental_state, double_blink, etc.)
            data: Message data
            
        Returns:
            Formatted message dictionary
        """
        return {
            "type": message_type,
            "data": data,
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat()
        }
    
    async def _send_to_client(self, websocket: websockets.WebSocketServerProtocol, message: Dict[str, Any]):
        """Send message to a specific client"""
        try:
            await websocket.send(json.dumps(message))
        except websockets.exceptions.ConnectionClosed:
            await self.unregister_client(websocket)
        except Exception as e:
            self.logger.error(f"Error sending message to client: {e}")
    
    async def broadcast_message(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        if not self.clients:
            return
            
        # Create a copy of clients to avoid modification during iteration
        clients_copy = self.clients.copy()
        
        for client in clients_copy:
            await self._send_to_client(client, message)
    
    async def send_mental_state(self, mental_state: str, average_ratio: float, 
                              individual_ratios: Dict[str, float]):
        """
        Send mental state update to all clients
        
        Args:
            mental_state: Overall mental state (focused/relaxed)
            average_ratio: Average alpha/beta ratio
            individual_ratios: Individual channel ratios
        """
        message = self._create_message(
            self.MESSAGE_TYPES["MENTAL_STATE"],
            {
                "mental_state": mental_state,
                "average_ratio": average_ratio,
                "individual_ratios": individual_ratios,
                "confidence": self._calculate_confidence(average_ratio)
            }
        )
        
        await self.broadcast_message(message)
        self.logger.debug(f"Sent mental state: {mental_state} (ratio: {average_ratio:.3f})")
    
    async def send_double_blink(self, double_blink_data: Dict[str, Any]):
        """
        Send double blink detection to all clients
        
        Args:
            double_blink_data: Double blink detection data
        """
        message = self._create_message(
            self.MESSAGE_TYPES["DOUBLE_BLINK"],
            {
                "time_interval": double_blink_data.get("time_interval", 0),
                "strength": double_blink_data.get("strength", 0),
                "detection_time": double_blink_data.get("detection_time", 0),
                "first_blink_amplitude": double_blink_data.get("first_blink", {}).get("amplitude", 0),
                "second_blink_amplitude": double_blink_data.get("second_blink", {}).get("amplitude", 0)
            }
        )
        
        await self.broadcast_message(message)
        self.logger.info(f"Sent double blink: {double_blink_data.get('time_interval', 0):.3f}s interval")
    
    async def send_single_blink(self, blink_data: Dict[str, Any]):
        """
        Send single blink detection to all clients
        
        Args:
            blink_data: Single blink detection data
        """
        message = self._create_message(
            self.MESSAGE_TYPES["SINGLE_BLINK"],
            {
                "amplitude": blink_data.get("amplitude", 0),
                "time": blink_data.get("time", 0),
                "channel": blink_data.get("channel", 0)
            }
        )
        
        await self.broadcast_message(message)
        self.logger.debug(f"Sent single blink: {blink_data.get('amplitude', 0):.1f}μV")
    
    async def send_system_status(self, status: str, message: str = ""):
        """
        Send system status update to all clients
        
        Args:
            status: System status (connected, disconnected, error, etc.)
            message: Additional status message
        """
        status_msg = self._create_message(
            self.MESSAGE_TYPES["SYSTEM_STATUS"],
            {
                "status": status,
                "message": message,
                "connected_clients": len(self.clients)
            }
        )
        
        await self.broadcast_message(status_msg)
        self.logger.info(f"Sent system status: {status}")
    
    def _calculate_confidence(self, ratio: float) -> float:
        """
        Calculate confidence score for mental state classification
        
        Args:
            ratio: Alpha/beta ratio
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Higher confidence for ratios further from threshold (0.6)
        distance_from_threshold = abs(ratio - 0.6)
        confidence = min(distance_from_threshold * 2, 1.0)  # Scale to 0-1
        return round(confidence, 2)
    
    async def start_server(self):
        """Start the WebSocket server"""
        self.logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        
        try:
            async with websockets.serve(self.handle_client, self.host, self.port):
                self.logger.info("WebSocket server started successfully")
                await self.send_system_status("server_started", "BCI WebSocket server is running")
                await asyncio.Future()  # Run forever
        except Exception as e:
            self.logger.error(f"Error starting WebSocket server: {e}")
            raise
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get server connection information"""
        return {
            "host": self.host,
            "port": self.port,
            "connected_clients": len(self.clients),
            "message_types": list(self.MESSAGE_TYPES.values())
        }


async def main():
    """
    Test function for WebSocket server
    """
    print("=== BCI WebSocket Server Test ===")
    
    # Create and start server
    server = BCIWebSocketServer(host="localhost", port=8765)
    
    try:
        await server.start_server()
    except KeyboardInterrupt:
        print("Server stopped by user")
    except Exception as e:
        print(f"Server error: {e}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the server
    asyncio.run(main())
