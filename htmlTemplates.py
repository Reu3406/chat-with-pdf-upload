css = """
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
"""

bot_template = """
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://corp.nhg.com.sg/HSOR/PublishingImages/gary.jpg" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
        <!--img src="https://www.seekpng.com/png/detail/118-1188982_doctor-clipart-clip-art-doctor-png.png" alt="Doctor Clipart - Clip Art Doctor Png@seekpng.com"-->
    </div>
    <div class="message">{{MSG}}</div>
</div>
"""

user_template = """
<div class="chat-message user">
    <div class="avatar">
        <img src="https://www.seekpng.com/png/detail/115-1150053_avatar-png-transparent-png-royalty-free-default-user.png" alt="Avatar Png Transparent Png Royalty Free - Default User Image Jpg@seekpng.com">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
"""
