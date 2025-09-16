"""
Simple WebSocket client to test BCI server connection
"""

import asyncio
import websockets
import json

async def test_client():
    """Test WebSocket client"""
    uri = "ws://localhost:8765"
    
    try:
        print("Connecting to WebSocket server...")
        async with websockets.connect(uri) as websocket:
            print("✅ Connected successfully!")
            
            message_count = 0
            
            # Listen for messages for 30 seconds (Python 3.9 compatible)
            start_time = asyncio.get_event_loop().time()
            timeout_duration = 30
            
            try:
                while True:
                    # Check if we've exceeded the timeout
                    if asyncio.get_event_loop().time() - start_time > timeout_duration:
                        print(f"⏰ Test completed after {timeout_duration} seconds. Received {message_count} messages.")
                        break
                    
                    try:
                        # Wait for message with a short timeout to allow checking the main timeout
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        
                        data = json.loads(message)
                        message_count += 1
                        
                        msg_type = data.get("type", "unknown")
                        print(f"📨 Message #{message_count}: {msg_type}")
                        
                        if msg_type == "system_status":
                            status = data.get("data", {}).get("status", "unknown")
                            print(f"   Status: {status}")
                        elif msg_type == "mental_state":
                            mental_state = data.get("data", {}).get("mental_state", "unknown")
                            ratio = data.get("data", {}).get("average_ratio", 0)
                            print(f"   Mental State: {mental_state} (ratio: {ratio:.3f})")
                        elif msg_type == "double_blink":
                            interval = data.get("data", {}).get("time_interval", 0)
                            print(f"   🎯 DOUBLE BLINK! Interval: {interval:.3f}s")
                        elif msg_type == "single_blink":
                            amplitude = data.get("data", {}).get("amplitude", 0)
                            print(f"   👁️ Single blink: {amplitude:.1f}μV")
                        
                    except asyncio.TimeoutError:
                        # Short timeout expired, continue loop to check main timeout
                        continue
                    except json.JSONDecodeError:
                        print(f"❌ Invalid JSON: {message}")
                    except Exception as e:
                        print(f"❌ Error processing message: {e}")
                        
            except websockets.exceptions.ConnectionClosed:
                print(f"🔌 Connection closed. Received {message_count} messages.")
                
    except ConnectionRefusedError:
        print("❌ Connection refused! Is the BCI server running?")
        print("💡 Run: python main.py")
    except Exception as e:
        print(f"❌ Connection error: {e}")

if __name__ == "__main__":
    print("=== BCI WebSocket Test Client ===")
    print("Testing connection to ws://localhost:8765")
    print()
    
    try:
        asyncio.run(test_client())
    except KeyboardInterrupt:
        print("\n👋 Test stopped by user")
