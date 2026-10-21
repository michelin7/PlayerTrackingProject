; Inno Setup script for Player Tracking (Windows).
; Per-user install: no admin password, no UAC prompt, fewer clicks.
#define AppName "Player Tracking"
#define AppVer  "1.0.0"

[Setup]
AppName={#AppName}
AppVersion={#AppVer}
AppPublisher=SUSU
DefaultDirName={localappdata}\Programs\PlayerTracking
DefaultGroupName=Player Tracking
PrivilegesRequired=lowest
DisableProgramGroupPage=yes
DisableDirPage=yes
OutputDir=..\dist
OutputBaseFilename=PlayerTracking-Setup
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\PlayerTracking.exe
Compression=lzma2/max
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern

[Languages]
Name: "en"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "..\dist\PlayerTracking\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "MicrosoftEdgeWebview2Setup.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall; Check: not WebView2Installed

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\PlayerTracking.exe"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\PlayerTracking.exe"

[Run]
; The window is WebView2; without the runtime the app would open blank.
Filename: "{tmp}\MicrosoftEdgeWebview2Setup.exe"; Parameters: "/silent /install"; \
    StatusMsg: "Installing the Microsoft WebView2 runtime..."; Check: not WebView2Installed
Filename: "{app}\PlayerTracking.exe"; Description: "Launch {#AppName}"; \
    Flags: nowait postinstall skipifsilent

[Code]
function WebView2Installed: Boolean;
var key: String;
begin
  key := 'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}';
  Result := RegKeyExists(HKLM, key) or
            RegKeyExists(HKCU, 'SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}');
end;
