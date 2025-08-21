import json
from channels.generic.websocket import AsyncWebsocketConsumer

class GestureConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        print("🔌 [Consumer] Attempting to add connection to 'gesture_group' ...")
        await self.channel_layer.group_add("gesture_group", self.channel_name)
        await self.accept()
        print(f"✅ [Consumer] WebSocket connection accepted! Channel: {self.channel_name}")

    async def disconnect(self, close_code):
        print(f"❌ [Consumer] Disconnecting. Close code: {close_code}. Removing channel: {self.channel_name} from 'gesture_group' ...")
        await self.channel_layer.group_discard("gesture_group", self.channel_name)
        print("❌ [Consumer] WebSocket disconnected.")

    async def receive(self, text_data):
        print(f"📥 [Consumer] Received from frontend: {text_data}")
        try:
            data = json.loads(text_data)
            gesture = data.get("gesture")
            if gesture:
                print(f"🖐️ [Consumer] Broadcasting gesture '{gesture}' to group 'gesture_group' ...")
                await self.channel_layer.group_send(
                    "gesture_group",
                    {
                        "type": "gesture_message",  # Must match the method name below exactly.
                        "gesture": gesture
                    }
                )
            else:
                print("⚠️ [Consumer] No gesture key found in received data!")
        except Exception as e:
            print(f"❗ [Consumer] Error processing received data: {e}")

    async def gesture_message(self, event):
        gesture = event.get("gesture")
        print(f"📥 [Consumer] Received gesture event from group: {gesture}")
        try:
            await self.send(text_data=json.dumps({
                "action": gesture
            }))
            print(f"📤 [Consumer] Sent to frontend: {{'action': '{gesture}'}}")
        except Exception as e:
            print(f"❗ [Consumer] Error sending message to frontend: {e}")
