; ============================================================
;  QuanSen — Inno Setup Installer Script  (onedir edition)
;  v1.0  by Amatra Sen
; ============================================================

#define AppName        "QuanSen"
#define AppVersion     "1.0"
#define AppPublisher   "Amatra Sen"
#define AppExeName     "QuanSen.exe"
#define AppIconFile    "quansen.ico"

[Setup]
AppId={{B7E2D4F1-9C3A-4E8B-A1D6-2F5E7B8C9D0E}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=installer_output
OutputBaseFilename=QuanSen_Setup_v{#AppVersion}
SetupIconFile={#AppIconFile}
WizardStyle=modern
WizardSizePercent=120
Compression=lzma2/ultra64
SolidCompression=yes
LZMAUseSeparateProcess=yes
UninstallDisplayIcon={app}\{#AppExeName}
UninstallDisplayName={#AppName} {#AppVersion}
ShowLanguageDialog=no
ArchitecturesInstallIn64BitMode=x64os

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon";   Description: "Create a &Desktop shortcut";    GroupDescription: "Shortcuts:"; Flags: checkedonce
Name: "startmenuicon"; Description: "Create a &Start Menu shortcut"; GroupDescription: "Shortcuts:"; Flags: checkedonce

[Files]
; ── Splash launcher (onefile) ─────────────────────────────────
Source: "dist\{#AppExeName}";             DestDir: "{app}";                Flags: ignoreversion

; ── App icon ──────────────────────────────────────────────────
Source: "{#AppIconFile}";                 DestDir: "{app}";                Flags: ignoreversion

; ── quansen_app onedir folder (entire contents, recursive) ───
Source: "dist\quansen_app\*";             DestDir: "{app}\quansen_app";    Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}";              Filename: "{app}\{#AppExeName}"; IconFilename: "{app}\{#AppIconFile}"; Tasks: startmenuicon
Name: "{group}\Uninstall {#AppName}";    Filename: "{uninstallexe}"
Name: "{userdesktop}\{#AppName}";        Filename: "{app}\{#AppExeName}"; IconFilename: "{app}\{#AppIconFile}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Launch {#AppName} now"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: files;     Name: "{app}\quansen_weights.csv"
Type: files;     Name: "{app}\quansen_report.txt"
Type: files;     Name: "{app}\quansen_asset_summary.csv"
Type: files;     Name: "{app}\quansen_frontier.csv"
