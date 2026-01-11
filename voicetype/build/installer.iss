; VoiceType - Inno Setup Installer Script
; Версия: 1.0.0
; Сборка: Полная с PyTorch

#define MyAppName "VoiceType"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "VoiceType Team"
#define MyAppURL "https://github.com/voicetype/voicetype"
#define MyAppExeName "VoiceType.exe"

[Setup]
; Основная информация
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}

; Директории установки
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes

; Выходной файл
OutputDir=..\installer
OutputBaseFilename=VoiceType-Setup-{#MyAppVersion}

; Сжатие
Compression=lzma2/ultra64
SolidCompression=yes
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=4

; Архитектура
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

; Внешний вид
WizardStyle=modern
SetupIconFile=..\resources\icons\app_icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}

; Права администратора не требуются для установки в пользовательскую папку
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

; Информация для отображения
AppCopyright=Copyright (c) 2025 {#MyAppPublisher}
VersionInfoVersion={#MyAppVersion}
VersionInfoCompany={#MyAppPublisher}
VersionInfoDescription={#MyAppName} - Голосовой ввод текста
VersionInfoProductName={#MyAppName}
VersionInfoProductVersion={#MyAppVersion}

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: checkedonce
Name: "autostart"; Description: "Запускать VoiceType при старте Windows"; GroupDescription: "Дополнительно:"; Flags: unchecked

[Files]
; Основные файлы приложения
Source: "dist\VoiceType\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; Примечание: модели включены в dist через PyInstaller

[Icons]
; Ярлыки в меню Пуск
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Comment: "Голосовой ввод текста"
Name: "{group}\Удалить {#MyAppName}"; Filename: "{uninstallexe}"

; Ярлык на рабочем столе
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon; Comment: "Голосовой ввод текста"

[Registry]
; Автозапуск при старте Windows
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "{#MyAppName}"; ValueData: """{app}\{#MyAppExeName}"""; Flags: uninsdeletevalue; Tasks: autostart

[Run]
; Запустить после установки
Filename: "{app}\{#MyAppExeName}"; Description: "Запустить {#MyAppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Удаление конфигурации при деинсталляции (опционально)
Type: filesandordirs; Name: "{userappdata}\{#MyAppName}"

[Code]
// Проверка что приложение не запущено перед установкой/удалением
function IsAppRunning(): Boolean;
var
  ResultCode: Integer;
begin
  Exec('tasklist', '/FI "IMAGENAME eq {#MyAppExeName}" /NH', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  Result := (ResultCode = 0);
end;

function InitializeSetup(): Boolean;
begin
  Result := True;

  // Проверяем, не запущено ли приложение
  if IsAppRunning() then
  begin
    if MsgBox('VoiceType сейчас запущен. Закройте приложение перед установкой.' + #13#10 + #13#10 +
              'Хотите попробовать закрыть автоматически?', mbConfirmation, MB_YESNO) = IDYES then
    begin
      Exec('taskkill', '/F /IM {#MyAppExeName}', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
      Sleep(1000);
    end
    else
    begin
      Result := False;
    end;
  end;
end;

function InitializeUninstall(): Boolean;
var
  ResultCode: Integer;
begin
  Result := True;

  // Закрываем приложение перед удалением
  Exec('taskkill', '/F /IM {#MyAppExeName}', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  Sleep(500);
end;
