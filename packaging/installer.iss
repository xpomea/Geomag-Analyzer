; Inno Setup 6/7 user-level installer script for GeoMag Analyzer (Windows x64).
; Non-admin installation into {localappdata}\Programs\GeoMagAnalyzer with runtime
; data (runtime.sqlite, user_cache/) stored in %LOCALAPPDATA%\GeoMagAnalyzer.

#define MyAppName "GeoMag Analyzer"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "ONU Thesis - Pc5/Pc6 Geomagnetic Research"
#define MyAppExeName "GeoMagAnalyzer.exe"

[Setup]
AppId={{8F5E3B7A-7C2E-4D1B-9A4C-52B1E0E4F701}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\GeoMagAnalyzer
DefaultGroupName={#MyAppName}
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=commandline
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
DisableProgramGroupPage=yes
OutputDir=..\dist
OutputBaseFilename=GeoMagAnalyzer_Setup
SetupIconFile=geomag_analyzer.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
WizardStyle=modern dynamic
Compression=lzma2/normal
SolidCompression=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\dist\GeoMagAnalyzer\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{#MyAppName} (Browser Mode)"; Filename: "{app}\{#MyAppExeName}"; Parameters: "--browser"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
